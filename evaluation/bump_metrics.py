import csv, hashlib, io, warnings
from pathlib import Path
from collections import Counter
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import fftconvolve
from scipy.optimize import linprog, linear_sum_assignment, OptimizeWarning


SHAPE = (232, 232)
PX, PY = 72.73999999999995, 72.13999999999993
RX, RY = 23.08859657150821, 25.770280355451852
BASE_PERCENTILE, TOP_WINDOW, SEARCH_RADIUS = 25, 7, 4
GT_DX, GT_DY, GT_RADIUS, GT_TOP_RADIUS = 72.0829, 71.9073, 23.9021, 6
MIN_POINTS, MAX_POINTS, MIN_SPAN = 64, 1500, .30
VERSION = f"pred_global{BASE_PERCENTILE}_l1_win{TOP_WINDOW}_search{SEARCH_RADIUS}_v2"
_CENTER_CACHE = {}


def phase(projection, pitch):
    p = gaussian_filter1d(projection.astype(float), 1., mode="reflect")
    if p.max() <= 0: return None
    edges = np.diff(np.r_[False, p >= max(.5, .43*p.max()), False].astype(np.int8))
    a, b = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = (a > 0) & (b < len(p)) & (b-a >= .12*pitch) & (b-a <= .88*pitch)
    centers = (a[keep]+b[keep]-1)/2
    if not len(centers): return None
    z = np.mean(np.exp(2j*np.pi*centers/pitch))
    return float(np.angle(z)*pitch/(2*np.pi) % pitch) if abs(z) >= 1e-8 else None


def grid(p, pitch, length):
    return p + np.arange(int(np.ceil(-p/pitch)), int(np.floor((length-1-p)/pitch))+1)*pitch


@lru_cache(maxsize=1)
def detection_template():
    hx, hy = int(np.ceil(1.4*RX)), int(np.ceil(1.4*RY))
    yy, xx = np.mgrid[-hy:hy+1, -hx:hx+1]; d = (xx/RX)**2+(yy/RY)**2
    core, ring = (d <= 1).astype(float), ((d >= 1.10**2) & (d <= 1.40**2)).astype(float)
    den = [np.maximum(fftconvolve(np.ones(SHAPE), k, mode="same"), 1) for k in (core, ring)]
    sx, sy = int(round(.12*PX)), int(round(.12*PY))
    offsets = sorted([(dx, dy) for dy in range(-sy, sy+1) for dx in range(-sx, sx+1)],
                     key=lambda q: (q[0]**2+q[1]**2, q[1], q[0]))
    return core, ring, den, np.array(offsets), sx, sy


def detect(image):
    low, high = np.percentile(image, (10, 95))
    if high-low <= 1e-8: raise ValueError("NO_HEIGHT_CONTRAST")
    norm = np.clip((image-low)/(high-low), 0, 1); binary = norm >= .59
    x0, y0 = phase(binary.sum(0), PX), phase(binary.sum(1), PY)
    if x0 is None or y0 is None: raise ValueError("NO_EDGE_SEED")

    h, w = image.shape; core, ring, den, offsets, sx, sy = detection_template()
    score = (fftconvolve(binary.astype(float), core, mode="same")/den[0]
             - fftconvolve(binary.astype(float), ring, mode="same")/den[1])
    xs, ys = grid(x0, PX, w), grid(y0, PY, h)
    ax, ay = xs[(xs-sx >= RX) & (xs+sx+RX <= w-1)], ys[(ys-sy >= RY) & (ys+sy+RY <= h-1)]

    if len(ax) and len(ay):
        xx, yy = np.meshgrid(ax, ay)
        qx, qy = xx.ravel()[None, :]+offsets[:, 0, None], yy.ravel()[None, :]+offsets[:, 1, None]
        values = map_coordinates(score, [qy.ravel(), qx.ravel()], order=1, mode="nearest")
        values = values.reshape(len(offsets), -1).mean(1)
        best, shift = -np.inf, (0, 0)
        for offset, value in zip(offsets, values):
            if value > best+1e-10: best, shift = value, offset
        x0, y0 = (x0+shift[0]) % PX, (y0+shift[1]) % PY

    rois = []
    for row, cy in enumerate(grid(y0, PY, h), 1):
        for col, cx in enumerate(grid(x0, PX, w), 1):
            if min(cx, cy, w-1-cx, h-1-cy) < 6: continue
            xa, xb = max(0, int(np.ceil(cx-PX/2))), min(w, int(np.ceil(cx+PX/2)))
            ya, yb = max(0, int(np.ceil(cy-PY/2))), min(h, int(np.ceil(cy+PY/2)))
            yy, xx = np.mgrid[ya:yb, xa:xb]; d = ((xx-cx)/RX)**2+((yy-cy)/RY)**2
            core, ring = d <= .55**2, (d >= 1.05**2) & (d <= 1.45**2)
            if core.sum() < 10 or ring.sum() < 10: continue
            z = norm[ya:yb, xa:xb]
            if z[core].mean()-np.median(z[ring]) < .12 or binary[ya:yb, xa:xb][core].mean() < .15:
                continue
            rois.append((f"R{row:02d}C{col:02d}", float(cx), float(cy)))
    return rois


def fit_base(image):
    h, w = image.shape; yy, xx = np.where(image <= np.percentile(image, BASE_PERCENTILE))
    if len(xx) < MIN_POINTS: raise ValueError("BASE_POINTS_SMALL")
    if np.ptp(xx) < MIN_SPAN*(w-1) or np.ptp(yy) < MIN_SPAN*(h-1): raise ValueError("BASE_SPAN_SMALL")
    if len(xx) > MAX_POINTS:
        take = np.linspace(0, len(xx)-1, MAX_POINTS, dtype=int); xx, yy = xx[take], yy[take]
    X = np.column_stack((xx/(w-1), yy/(h-1), np.ones(len(xx))))
    if np.linalg.matrix_rank(X) < 3: raise ValueError("BASE_RANK_SMALL")
    offset = np.median(image[yy, xx]); z = image[yy, xx]-offset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        result = linprog(z, A_eq=X.T, b_eq=np.zeros(3), bounds=(-1, 1),
                         method="highs-ds", options={"threads": 1})

    beta = getattr(getattr(result, "eqlin", None), "marginals", None)
    if not result.success or beta is None: raise ValueError("L1_SOLVER_FAILED")
    if not np.isfinite(beta).all() or not np.isclose(np.abs(X@beta-z).sum(), -result.fun, rtol=1e-7, atol=1e-7):
        raise ValueError("L1_OPTIMALITY_FAILED")
    beta = np.array(beta); beta[2] += offset
    return beta


def measure(image, rois):
    beta = fit_base(image)
    h, w = image.shape; win = TOP_WINDOW; half = (win-1)/2
    integral = np.pad(image.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    means = (integral[win:, win:] - integral[:-win, win:]
             - integral[win:, :-win] + integral[:-win, :-win]) / win**2
    heights = []

    for _, cx, cy in rois:
        if win % 2:
            ax, ay = round(cx)-win//2, round(cy)-win//2
        else:
            ax, ay = int(np.floor(cx))-win//2+1, int(np.floor(cy))-win//2+1
        x, y = [v.ravel() for v in np.meshgrid(np.arange(ax-SEARCH_RADIUS, ax+SEARCH_RADIUS+1),
                                              np.arange(ay-SEARCH_RADIUS, ay+SEARCH_RADIUS+1))]
        keep = (x >= 0) & (y >= 0) & (x+win <= w) & (y+win <= h); x, y = x[keep], y[keep]
        if not len(x): heights.append(np.nan); continue
        base = beta[0]*(x+half)/(w-1) + beta[1]*(y+half)/(h-1) + beta[2]
        heights.append(float(np.max(means[y, x]-base)))
    return np.array(heights)


def gt_centers(gt):
    h, w = gt.shape; r = int(np.ceil(GT_RADIUS)); binary = ((gt == 0) | (gt >= 5)).astype(float)

    def guess(pr):
        a = np.flatnonzero((pr[:-1] < .5) & (pr[1:] >= .5))+1
        b = np.flatnonzero((pr[:-1] >= .5) & (pr[1:] < .5))+1
        centers = [(v+b[b > v][0])/2 for v in a if np.any(b > v)]
        return centers[0] if centers else float(np.argmax(pr))

    def positions(c, p, length):
        return np.r_[np.arange(c, -r, -p)[::-1], np.arange(c+p, length+r, p)]

    def selectors(c, p, length):
        M = np.zeros((61, length))
        for i, d in enumerate(range(-30, 31)):
            pos = np.rint(positions(c+d, p, length)).astype(int)
            pos = pos[(pos >= r) & (pos+r < length)]; M[i, pos] = 1
        return M

    x0, y0 = guess(binary.sum(0)), guess(binary.sum(1)); yy, xx = np.ogrid[-r:r+1, -r:r+1]
    score = np.rint(fftconvolve(binary, (xx*xx+yy*yy <= GT_RADIUS**2).astype(float), mode="same"))
    X, Y = selectors(x0, GT_DX, w), selectors(y0, GT_DY, h)
    count = Y.sum(1)[:, None]*X.sum(1)[None, :]
    values = np.divide(Y@score@X.T, count, out=np.full((61, 61), -np.inf), where=count > 0)
    if not np.isfinite(values).any(): raise ValueError("GT_TEMPLATE_FAILED")

    iy, ix = np.unravel_index(np.argmax(values), values.shape)
    xs, ys = np.rint(positions(x0+ix-30, GT_DX, w)), np.rint(positions(y0+iy-30, GT_DY, h))
    xs = xs[(xs >= GT_TOP_RADIUS) & (xs+GT_TOP_RADIUS < w)]
    ys = ys[(ys >= GT_TOP_RADIUS) & (ys+GT_TOP_RADIUS < h)]
    return np.array([(x, y) for y in ys for x in xs], dtype=float).reshape(-1, 2)


@lru_cache(maxsize=2048)
def reference(folder, stem):
    heights = None
    try:
        with (Path(folder)/"height"/f"{stem}.csv").open(newline="", encoding="utf-8-sig") as f:
            rows = sorted(csv.DictReader(f), key=lambda a: int(a["idx"]))
        if not rows or [int(a["idx"]) for a in rows] != list(range(1, len(rows)+1)):
            raise ValueError("INVALID_GT_IDX")
        heights = np.array([float(a["final_h"]) for a in rows])
        if not np.isfinite(heights).all(): raise ValueError("INVALID_FINAL_H")

        blob = (Path(folder)/"interfero"/f"{stem}.csv").read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest not in _CENTER_CACHE:
            gt = np.loadtxt(io.BytesIO(blob), delimiter=",")
            if gt.ndim != 2 or not np.isfinite(gt).all(): raise ValueError("INVALID_GT_MAP")
            if len(_CENTER_CACHE) >= 2048: _CENTER_CACHE.clear()
            _CENTER_CACHE[digest] = (gt_centers(gt), gt.shape)

        native, shape = _CENTER_CACHE[digest]
        if len(native) != len(heights): return heights, None, None, "GT_GRID_COUNT_MISMATCH"
        centers = (native+.5)*np.array([SHAPE[1]/shape[1], SHAPE[0]/shape[0]])-.5
        limit = .25*min(GT_DX*SHAPE[1]/shape[1], GT_DY*SHAPE[0]/shape[0])
        return heights, centers, limit, ""
    except Exception as exc:
        return heights, None, None, f"GT_UNAVAILABLE: {exc}"


def evaluate_map(job):
    scan, stem, image, gt_folder = job
    rois, values, refs, error = [], np.array([]), {}, ""
    try:
        image = np.asarray(image, dtype=float)
        if image.shape != SHAPE or not np.isfinite(image).all(): raise ValueError("INVALID_PRED")
        rois = detect(image)
        if not rois: raise ValueError("NO_DETECTIONS")
        values = measure(image, rois)
        if not np.isfinite(values).all(): raise ValueError("NONFINITE_HEIGHT")
    except Exception as exc:
        error = str(exc); values = np.full(len(rois), np.nan)

    gt, centers, limit, gt_error = reference(str(gt_folder), stem)
    expected = len(gt) if gt is not None else None
    status = "MEASUREMENT_FAILED" if error else "MATCH"
    if not error:
        if gt_error: status = gt_error.split(":")[0]
        elif len(rois) != expected: status = "COUNT_MISMATCH"
        else:
            xy = np.array([(cx, cy) for _, cx, cy in rois])
            distance = np.linalg.norm(xy[:, None, :]-centers[None, :, :], axis=2)
            ii, jj = linear_sum_assignment(distance)
            if np.any(distance[ii, jj] > limit): status = "POSITION_MISMATCH"
            else: refs = {int(i): (int(j+1), float(gt[j])) for i, j in zip(ii, jj)}

    details = [[scan, stem, gid, cx, cy, float(values[j]), *refs.get(j, ("", np.nan))]
               for j, (gid, cx, cy) in enumerate(rois)]
    info = [scan, stem, len(rois), int(np.isfinite(values).sum()), expected, len(refs), status, error or gt_error]
    return details, info


class BumpEvaluator:
    def __init__(self, scans, gt_root, output_dir, workers=4, repeated=True):
        if workers < 1: raise ValueError("eval_workers must be positive")
        self.scans, self.gt_root, self.output_dir = tuple(scans), Path(gt_root), Path(output_dir)
        self.workers, self.repeated, self.pool = workers, repeated, None
        self.details, self.maps, self.seen = [], [], set()

    def __enter__(self):
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"))
        return self

    def __exit__(self, *exc):
        if self.pool is not None: self.pool.shutdown(wait=True, cancel_futures=True)

    def update_batch(self, scan, stems, predictions):
        if len(stems) != len(predictions): raise ValueError("Batch length mismatch")
        folder = self.gt_root/str(scan) if self.repeated else self.gt_root
        jobs = []
        for stem, image in zip(stems, predictions):
            if scan not in self.scans or (scan, stem) in self.seen:
                raise ValueError(f"Duplicate/invalid sample: {scan}/{stem}")
            self.seen.add((scan, stem)); jobs.append((scan, stem, image, str(folder)))
        results = self.pool.map(evaluate_map, jobs) if self.pool is not None else map(evaluate_map, jobs)
        for details, info in results:
            self.details.extend(details); self.maps.append(info)

    def finish(self):
        counts, statuses, known, bumps = Counter(), Counter(), set(), {}
        for scan, stem, detected, measured, expected, matched, status, error in self.maps:
            counts.update(maps=1, detected=detected, measured=measured, expected=expected or 0,
                          matched=matched, unknown_gt=int(expected is None))
            statuses[status] += 1
            if expected is not None: known.update((stem, i) for i in range(1, expected+1))

        scan_index = {s: i for i, s in enumerate(self.scans)}
        for scan, stem, gid, cx, cy, value, idx, target in self.details:
            if idx == "" or not np.isfinite([value, target]).all(): continue
            key = (stem, idx)
            if key not in bumps: bumps[key] = np.full((len(self.scans), 2), np.nan)
            bumps[key][scan_index[scan]] = value, target

        common = {k: v for k, v in bumps.items() if np.isfinite(v).all()}
        n = len(common); data = np.stack(list(common.values())) if n else np.empty((0, len(self.scans), 2))
        pred, gt = data[:, :, 0], data[:, :, 1]
        eb = float(np.abs(pred-gt).mean()) if n else np.nan
        sb = float(pred.std(1, ddof=0).mean()) if n and len(self.scans) > 1 else np.nan
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for name, columns, rows in (
            ("height_maps.csv", ["scan", "stem", "detected", "measured", "gt_expected", "matched", "status", "error"], self.maps),
            ("height_details.csv", ["scan", "stem", "pred_id", "cx", "cy", "pred_um", "gt_idx", "gt_um", "complete_scans"],
             [r+[int((r[1], r[6]) in common)] for r in self.details])):
            with (self.output_dir/name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f); writer.writerow(columns)
                writer.writerows(sorted(rows, key=lambda r: (r[0], r[1])))

        lines = ["BUMP HEIGHT / "+VERSION, "Unit: 10^-2 um; detail CSV heights: um.",
                 f"Scans: {list(self.scans)} | GT_ROOT: {self.gt_root}",
                 f"Base: global P{BASE_PERCENTILE}, L1, max {MAX_POINTS}, min {MIN_POINTS}, span {MIN_SPAN}, rank 3",
                 f"Top: {TOP_WINDOW}x{TOP_WINDOW}, search +/-{SEARCH_RADIUS} px; maximum plane-corrected mean",
                 "Detection: P10-P95 / binary 0.59 / projection 0.43 / grid refinement / boundary 6 px",
                 f"Maps: {counts['maps']} | Detected: {counts['detected']} | Measured: {counts['measured']}",
                 f"Map status: {dict(sorted(statuses.items()))}",
                 f"Known GT pairs: {counts['expected']} | Unknown GT count maps: {counts['unknown_gt']}",
                 f"Matched pairs before complete-scan filtering: {counts['matched']}",
                 f"Common bumps: {n} | Evaluated pairs: {n*len(self.scans)}",
                 f"Excluded known GT pairs: {counts['expected']-n*len(self.scans)}",
                 f"Known unique GT bumps: {len(known)} | Excluded: {len(known)-n}",
                 f"E_b: {100*eb:.6f} | sigma_b: {100*sb:.6f}",
                 f"Mean GT: {100*gt.mean() if n else np.nan:.6f} | Mean prediction: {100*pred.mean() if n else np.nan:.6f}",
                 "", f"{'Scan':>6} {'N':>8} {'GT mean':>16} {'Pred mean':>16} {'MAE':>16}"]

        for i, scan in enumerate(self.scans):
            g, p, e = (gt[:, i].mean(), pred[:, i].mean(), np.abs(pred[:, i]-gt[:, i]).mean()) if n else (np.nan,)*3
            lines.append(f"{scan:6} {n:8} {100*g:16.6f} {100*p:16.6f} {100*e:16.6f}")
        lines += ["", "E_b = mean_s mean_i abs(pred-GT); sigma_b = mean_i STD_s(pred), ddof=0.",
                  "GT final_h unchanged; GT used only for matching/evaluation.",
                  "All rows use the same complete-scan cohort. Repeatability requires at least two scans.",
                  "Only samples visited in this test are counted. Existing prediction CSVs are not reused.",
                  "Compare cohort IDs across methods, not only counts."]
        (self.output_dir/"height_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
        return {"E_b": eb, "sigma_b": sb}