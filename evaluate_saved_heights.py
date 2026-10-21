
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

import csv, hashlib, io, warnings, inspect, json, re, shlex, subprocess, sys
from pathlib import Path
from collections import Counter
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import fftconvolve
from scipy.optimize import linprog, linear_sum_assignment, OptimizeWarning
from tqdm import tqdm

# ==================== 설정 ====================
FORCE = False

REMOTE_SERVERS = [
    dict(name="28", host="203.255.176.28", port=42362, base_dir="/home/dspl/CY/Bump/Model",
         gt_root="/home/dspl/CY/Bump/Dataset/coin_bump_dataset_2/repeat_test",
         python="/home/dspl/CY/dspl/bin/python", workers=8),
    dict(name="97", host="203.255.177.97", port=42362, base_dir="/home/dspl/CY/Bump/Model",
         gt_root="/home/dspl/CY/Bump/Dataset/coin_bump_dataset_2/repeat_test",
         python="python3", workers=12),
    dict(name="245", host="203.255.177.245", port=22, base_dir="/home/dspl/CY/Bump/Model",
         gt_root="/home/dspl/CY/Bump/Dataset/coin_bump_dataset_2/repeat_test",
         python="/home/dspl/CY/dspl/bin/python", workers=8),
]

MODEL_ROOT = Path("/home/dspl/CY/Bump/Model")
GT_ROOT = Path("/home/dspl/CY/Bump/Dataset/coin_bump_dataset_2/repeat_test")
SCANS, WORKERS = tuple(range(1, 11)), 20  # Local 20 workers, HDD 직접 읽기

PX, PY = 72.73999999999995, 72.13999999999993
RX, RY = 23.08859657150821, 25.770280355451852
TOP_WINDOW, SEARCH_RADIUS = 7, 4
GT_DX, GT_DY, GT_RADIUS, GT_TOP_RADIUS = 72.0829, 71.9073, 23.9021, 6
MIN_POINTS, MAX_POINTS, MIN_SPAN = 64, 1500, .30
VERSION = "pred_global10_l1_win7_search4_v1"
_CENTER_CACHE = {}

# ==================== Pred 중심 검출 ====================
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
    den = [np.maximum(fftconvolve(np.ones((232, 232)), k, mode="same"), 1) for k in (core, ring)]
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
    score = fftconvolve(binary.astype(float), core, mode="same")/den[0] - fftconvolve(binary.astype(float), ring, mode="same")/den[1]
    xs, ys = grid(x0, PX, w), grid(y0, PY, h)
    ax, ay = xs[(xs-sx >= RX) & (xs+sx+RX <= w-1)], ys[(ys-sy >= RY) & (ys+sy+RY <= h-1)]

    if len(ax) and len(ay):
        xx, yy = np.meshgrid(ax, ay)
        qx, qy = xx.ravel()[None, :]+offsets[:, 0, None], yy.ravel()[None, :]+offsets[:, 1, None]
        values = map_coordinates(score, [qy.ravel(), qx.ravel()], order=1, mode="nearest").reshape(len(offsets), -1).mean(1)
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
            if z[core].mean()-np.median(z[ring]) < .12 or binary[ya:yb, xa:xb][core].mean() < .15: continue
            rois.append((f"R{row:02d}C{col:02d}", float(cx), float(cy)))
    return rois

# ==================== Global L1 평면 + 7x7 Top ====================
def fit_base(image):
    h, w = image.shape; yy, xx = np.where(image <= np.percentile(image, 25))
    if len(xx) < MIN_POINTS: raise ValueError("BASE_POINTS_SMALL")
    if np.ptp(xx) < MIN_SPAN*(w-1) or np.ptp(yy) < MIN_SPAN*(h-1): raise ValueError("BASE_SPAN_SMALL")
    if len(xx) > MAX_POINTS:
        take = np.linspace(0, len(xx)-1, MAX_POINTS, dtype=int); xx, yy = xx[take], yy[take]
    X = np.column_stack((xx/(w-1), yy/(h-1), np.ones(len(xx))))
    if np.linalg.matrix_rank(X) < 3: raise ValueError("BASE_RANK_SMALL")
    offset = np.median(image[yy, xx]); z = image[yy, xx]-offset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        result = linprog(z, A_eq=X.T, b_eq=np.zeros(3), bounds=(-1, 1), method="highs-ds", options={"threads": 1})

    beta = getattr(getattr(result, "eqlin", None), "marginals", None)
    if not result.success or beta is None: raise ValueError("L1_SOLVER_FAILED")
    if not np.isfinite(beta).all() or not np.isclose(np.abs(X@beta-z).sum(), -result.fun, rtol=1e-7, atol=1e-7):
        raise ValueError("L1_OPTIMALITY_FAILED")
    beta = np.array(beta); beta[2] += offset
    return beta

def measure(image, rois):
    beta = fit_base(image)
    h, w = image.shape; win = TOP_WINDOW; half = (win - 1) / 2
    integral = np.pad(image.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    means = (integral[win:, win:] - integral[:-win, win:] - integral[win:, :-win] + integral[:-win, :-win]) / (win * win)
    heights = []

    for _, cx, cy in rois:
        if win % 2:
            ax, ay = round(cx) - win // 2, round(cy) - win // 2
        else:
            ax, ay = int(np.floor(cx)) - win // 2 + 1, int(np.floor(cy)) - win // 2 + 1

        x, y = [v.ravel() for v in np.meshgrid(np.arange(ax-SEARCH_RADIUS, ax+SEARCH_RADIUS+1),
                                                np.arange(ay-SEARCH_RADIUS, ay+SEARCH_RADIUS+1))]
        keep = (x >= 0) & (y >= 0) & (x + win <= w) & (y + win <= h)
        x, y = x[keep], y[keep]
        if len(x) == 0: heights.append(np.nan); continue

        base = beta[0]*(x+half)/(w-1) + beta[1]*(y+half)/(h-1) + beta[2]
        heights.append(float(np.max(means[y, x] - base)))

    return np.array(heights)

# ==================== GT 위치 복원 ====================
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
    X, Y = selectors(x0, GT_DX, w), selectors(y0, GT_DY, h); count = Y.sum(1)[:, None]*X.sum(1)[None, :]
    values = np.divide(Y@score@X.T, count, out=np.full((61, 61), -np.inf), where=count > 0)
    if not np.isfinite(values).any(): raise ValueError("GT_TEMPLATE_FAILED")

    iy, ix = np.unravel_index(np.argmax(values), values.shape)
    xs, ys = np.rint(positions(x0+ix-30, GT_DX, w)), np.rint(positions(y0+iy-30, GT_DY, h))
    xs, ys = xs[(xs >= GT_TOP_RADIUS) & (xs+GT_TOP_RADIUS < w)], ys[(ys >= GT_TOP_RADIUS) & (ys+GT_TOP_RADIUS < h)]
    return np.array([(x, y) for y in ys for x in xs], dtype=float).reshape(-1, 2)

@lru_cache(maxsize=2048)
def reference(folder, stem):
    heights = None
    try:
        with (Path(folder)/"height"/f"{stem}.csv").open(newline="", encoding="utf-8-sig") as f:
            rows = sorted(csv.DictReader(f), key=lambda a: int(a["idx"]))
        if not rows or [int(a["idx"]) for a in rows] != list(range(1, len(rows)+1)): raise ValueError("INVALID_GT_IDX")
        heights = np.array([float(a["final_h"]) for a in rows])
        if not np.isfinite(heights).all(): raise ValueError("INVALID_FINAL_H")

        blob = (Path(folder)/"interfero"/f"{stem}.csv").read_bytes(); digest = hashlib.sha256(blob).hexdigest()
        if digest not in _CENTER_CACHE:
            gt = np.loadtxt(io.BytesIO(blob), delimiter=",")
            if gt.ndim != 2 or not np.isfinite(gt).all(): raise ValueError("INVALID_GT_MAP")
            if len(_CENTER_CACHE) >= 2048: _CENTER_CACHE.clear()
            _CENTER_CACHE[digest] = (gt_centers(gt), gt.shape)

        native, shape = _CENTER_CACHE[digest]
        if len(native) != len(heights): return heights, None, None, "GT_GRID_COUNT_MISMATCH"
        centers = (native+.5)*np.array([232/shape[1], 232/shape[0]])-.5
        limit = .25*min(GT_DX*232/shape[1], GT_DY*232/shape[0])
        return heights, centers, limit, ""
    except Exception as exc:
        return heights, None, None, f"GT_UNAVAILABLE: {exc}"

# ==================== 공통 GT Reference Cache ====================
GT_CACHE_VERSION = "gt_reference_v1"
_GT_LOOKUP = {}

def gt_cache_signature(stems):
    digest = hashlib.sha256()
    config = (GT_CACHE_VERSION, str(GT_ROOT.resolve()), SCANS, GT_DX, GT_DY, GT_RADIUS, GT_TOP_RADIUS)
    digest.update(repr(config).encode())

    for scan in SCANS:
        folder = GT_ROOT / str(scan)
        for stem in sorted(stems):
            for subdir in ("height", "interfero"):
                path = folder / subdir / f"{stem}.csv"
                try:
                    st = path.stat(); info = f"{st.st_size}:{st.st_mtime_ns}"
                except FileNotFoundError:
                    info = "MISSING"
                digest.update(f"{scan}/{subdir}/{stem}:{info}\n".encode())

    return digest.hexdigest()

def build_gt_stem(job):
    gt_root, stem = job; records = {}
    for scan in SCANS:
        folder = str(Path(gt_root) / str(scan))
        heights, centers, limit, error = reference(folder, stem)
        records[f"{scan}/{stem}"] = [None if heights is None else heights.tolist(),
                                     None if centers is None else centers.tolist(),
                                     None if limit is None else float(limit), error]
    return records

def prepare_gt_cache(stems):
    cache_path = MODEL_ROOT / "Height_SUMMARY" / "gt_reference_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signature = gt_cache_signature(stems)

    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("signature") == signature:
                print(f"[GT CACHE] Reusing {len(payload['records'])} references: {cache_path}", flush=True)
                return cache_path
        except (OSError, ValueError, KeyError, TypeError):
            pass

    print(f"[GT CACHE] Building references for {len(stems)} unique ROI names...", flush=True)
    records = {}; jobs = [(str(GT_ROOT), stem) for stem in sorted(stems)]

    with ProcessPoolExecutor(max_workers=min(WORKERS, 4)) as pool:
        for data in tqdm(pool.map(build_gt_stem, jobs, chunksize=2), total=len(jobs), desc="GT cache build"):
            records.update(data)

    payload = {"signature": signature, "version": GT_CACHE_VERSION, "records": records}
    temp = cache_path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(temp, cache_path)
    print(f"[GT CACHE] Saved {len(records)} references: {cache_path}", flush=True)
    return cache_path

def init_gt_cache(cache_path):
    global _GT_LOOKUP
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    _GT_LOOKUP = {
        key: (None if heights is None else np.asarray(heights, dtype=float),
              None if centers is None else np.asarray(centers, dtype=float), limit, error)
        for key, (heights, centers, limit, error) in payload["records"].items()
    }

def get_gt_reference(scan, stem, gt_root):
    key = f"{scan}/{stem}"
    if key in _GT_LOOKUP: return _GT_LOOKUP[key]
    return reference(str(Path(gt_root) / str(scan)), stem)

# ==================== ROI별 10개 Scan 처리 ====================
def process_roi(job):
    repeat, gt_root, stem = job; repeat, gt_root = Path(repeat), Path(gt_root)
    details, maps, bumps, known_ids = [], [], {}, set()

    for s, scan in enumerate(SCANS):
        rois, heights, refs, error = [], np.array([]), {}, ""
        path = repeat / str(scan) / "map/pred" / f"{stem}.csv"

        try:
            if not path.is_file(): raise ValueError("PRED_MISSING")
            image = np.loadtxt(path, delimiter=",")
            if image.shape != (232, 232) or not np.isfinite(image).all(): raise ValueError("INVALID_PRED_SHAPE_OR_VALUES")
            rois = detect(image)
            if not rois: raise ValueError("NO_DETECTIONS")
            heights = measure(image, rois)
        except Exception as exc:
            error = str(exc); heights = np.full(len(rois), np.nan)

        gt, centers, limit, gt_error = get_gt_reference(scan, stem, gt_root)
        expected = len(gt) if gt is not None else None
        if expected is not None: known_ids.update(range(1, expected+1))
        status = "MEASUREMENT_FAILED" if error else "MATCH"

        if not error:
            if gt_error: status = gt_error.split(":")[0]
            elif expected != len(rois): status = "COUNT_MISMATCH"
            else:
                xy = np.array([(r[1], r[2]) for r in rois])
                distance = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=2)
                ii, jj = linear_sum_assignment(distance)
                if np.any(distance[ii, jj] > limit): status = "POSITION_MISMATCH"
                else: refs = {int(i): (int(j+1), float(gt[j])) for i, j in zip(ii, jj)}

        for j, (gid, cx, cy) in enumerate(rois):
            idx, target = refs.get(j, ("", np.nan)); value = float(heights[j])
            details.append([scan, stem, gid, cx, cy, value, idx, target])
            if idx != "" and np.isfinite(value):
                if idx not in bumps: bumps[idx] = np.full((len(SCANS), 2), np.nan)
                bumps[idx][s] = value, target

        maps.append([scan, stem, len(rois), int(np.isfinite(heights).sum()), expected, len(refs), status, error or gt_error])

    common = {idx: a for idx, a in bumps.items() if np.isfinite(a).all()}
    for row in details: row.append(int(row[6] in common))
    data = np.stack(list(common.values())) if common else np.empty((0, len(SCANS), 2))
    pred, gt = data[:, :, 0], data[:, :, 1]

    return details, maps, len(common), len(known_ids), pred.sum(0), gt.sum(0), np.abs(pred-gt).sum(0), pred.std(1, ddof=0).sum()

# ==================== 저장 및 모델별 집계 ====================
def write_text(path, lines):
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def evaluate(pool, repeat, stems):
    counts, statuses = Counter(), Counter()
    ps, gs, es = np.zeros((3, len(SCANS))); ss = 0.
    paths = [repeat/"height_details.csv", repeat/"height_maps.csv"]

    with paths[0].with_suffix(".tmp").open("w", newline="", encoding="utf-8") as a, paths[1].with_suffix(".tmp").open("w", newline="", encoding="utf-8") as b:
        details, maps = csv.writer(a), csv.writer(b)
        details.writerow(["scan", "stem", "pred_id", "cx", "cy", "pred_um", "gt_idx", "gt_um", "complete10"])
        maps.writerow(["scan", "stem", "detected", "measured", "gt_expected", "matched", "status", "error"])

        futures = [pool.submit(process_roi, (str(repeat), str(GT_ROOT), stem)) for stem in stems]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{repeat.parent.parent.name}/{repeat.parent.name}"):
            d, m, n, known, p, g, e, std = future.result()
            details.writerows(d); maps.writerows(m)
            counts.update(common=n, known_unique=known); ps += p; gs += g; es += e; ss += std

            for row in m:
                _, _, detected, measured, expected, matched, status, _ = row
                counts.update(maps=1, detected=detected, measured=measured, matched=matched,
                              expected=expected or 0, unknown_gt=int(expected is None))
                statuses[status] += 1

    for path in paths: os.replace(path.with_suffix(".tmp"), path)

    n = counts["common"]; scale = 100/n if n else np.nan
    eb, sb = float(es.mean()*scale), float(ss*scale)

    lines = [
        "="*96, "BUMP HEIGHT / " + VERSION,
        f"Model: {repeat.parent.parent.name} | {repeat.parent.name}",
        "Unit: 10^-2 um (all height/MAE/STD columns); CSV heights: um.",
        "Base: global bottom 25% candidates / L1 / max 1500 points | Top: 7x7 / +/-4 px / plane-corrected maximum",
        "Detection: P10-P95 normalization / binary=0.59 / projection=0.43 / boundary=6 px",
        f"GT_ROOT: {GT_ROOT}",
        f"Maps: {counts['maps']} | Detected: {counts['detected']} | Measured: {counts['measured']}",
        f"Map status: {dict(sorted(statuses.items()))}",
        f"Known GT pairs: {counts['expected']} | Unknown GT count maps: {counts['unknown_gt']}",
        f"Matched pairs before complete-10 filtering: {counts['matched']}",
        f"Common bumps: {n} | Evaluated pairs: {n*len(SCANS)} | Excluded known GT pairs: {counts['expected']-n*len(SCANS)}",
        f"Known unique GT bumps: {counts['known_unique']} | Excluded: {counts['known_unique']-n}",
        "-"*96, f"E_b: {eb:.6f} | sigma_b: {sb:.6f}",
        f"Mean GT: {gs.mean()*scale:.6f} | Mean prediction: {ps.mean()*scale:.6f}",
        "-"*96, f"{'Scan':>6} {'N':>8} {'GT mean':>16} {'Pred mean':>16} {'MAE':>16}"
    ]

    for i, scan in enumerate(SCANS):
        lines.append(f"{scan:6d} {n:8d} {gs[i]*scale:16.6f} {ps[i]*scale:16.6f} {es[i]*scale:16.6f}")

    lines += [
        "", "E_b = mean_s mean_i abs(pred-GT); sigma_b = mean_i STD_s(pred), ddof=0.",
        "All table rows use the same complete-10 cohort. Different models may have different cohorts.",
        "GT final_h unchanged. GT used only for matching/evaluation. All predicted heights recomputed.",
        "height_details.csv: centers/heights + complete10 flag; height_maps.csv: exclusions/errors."
    ]

    write_text(repeat/"height_summary.txt", lines)
    print(f"  N={n} | E_b={eb:.4f} | sigma_b={sb:.4f} | {repeat/'height_summary.txt'}", flush=True)
    return f"{repeat.parent.parent.name:<70} {repeat.parent.name:12} {n:8d} {eb:12.6f} {sb:12.6f}"

# ==================== 완료 여부 ====================
def summary_values(text):
    if not re.search(r"(?m)^BUMP HEIGHT / " + re.escape(VERSION) + r"[ \t]*$", text): return None
    if "Unit: 10^-2 um" not in text: return None
    pair = re.search(r"(?m)^E_b:\s*(\S+)\s*\|\s*sigma_b:\s*(\S+)", text)
    n = re.search(r"Common bumps:\s*(\d+)", text)

    try:
        eb, sb, count = float(pair[1]), float(pair[2]), int(n[1])
        return (count, eb, sb) if count > 0 and np.isfinite([eb, sb]).all() and min(eb, sb) >= 0 else None
    except (TypeError, ValueError):
        return None

def completed(repeat):
    if FORCE: return False
    files = [repeat/name for name in ("height_summary.txt", "height_details.csv", "height_maps.csv")]
    if not all(p.is_file() and p.stat().st_size > 0 for p in files): return False
    return summary_values(files[0].read_text(encoding="utf-8")) is not None

# ==================== 현재 서버 평가: Direct HDD ====================
def run_local():
    if not MODEL_ROOT.is_dir(): raise FileNotFoundError(MODEL_ROOT)
    if not GT_ROOT.is_dir(): raise FileNotFoundError(f"GT_ROOT 확인: {GT_ROOT}")

    gt_stems = {p.stem for s in SCANS for p in (GT_ROOT/str(s)/"height").glob("*.csv")}
    jobs = []

    for model in sorted(p for p in MODEL_ROOT.iterdir() if p.is_dir() and p.name != "Height_SUMMARY"):
        for mode in ("model_final", "model_mini"):
            repeat = model / mode / "repeat_test"
            if not repeat.is_dir(): continue
            if completed(repeat):
                jobs.append((model.name, mode, repeat, None))
                continue

            stems = {p.stem for s in SCANS for p in (repeat/str(s)/"map/pred").glob("*.csv")}
            if not stems:
                print(f"NO PRED / SKIP: {model.name}/{mode}", flush=True)
                continue

            jobs.append((model.name, mode, repeat, sorted(stems | gt_stems)))

    lines = [
        f"BUMP HEIGHT / {VERSION} / unit: 10^-2 um",
        f"{'Model':<70} {'CP':12} {'N':>8} {'E_b':>12} {'sigma_b':>12}"
    ]

    if not jobs:
        print("No model evaluation jobs.", flush=True)
        return

    pending = sum(stems is not None for _, _, _, stems in jobs)
    print(f"[JOBS] Total={len(jobs)} Pending={pending} Completed={len(jobs)-pending}", flush=True)
    print(f"[DIRECT HDD] Workers={WORKERS} | RAM staging=OFF", flush=True)

    def run_jobs(pool):
        for model_name, mode, repeat, stems in jobs:
            if stems is None:
                n, eb, sb = summary_values((repeat/"height_summary.txt").read_text(encoding="utf-8"))
                print(f"DONE / SKIP: {model_name}/{mode}", flush=True)
                line = f"{model_name:<70} {mode:12} {n:8d} {eb:12.6f} {sb:12.6f}"
            else:
                # RAM 복사 없이 원본 HDD에서 직접 평가
                print(f"[EVALUATING] {model_name}/{mode} | workers={WORKERS} | ROIs={len(stems)}", flush=True)
                line = evaluate(pool, repeat, stems)

            lines.append(line)
            write_text(MODEL_ROOT/"height_summary_all.txt", lines)

    if pending:
        cache_path = prepare_gt_cache(gt_stems)
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_gt_cache, initargs=(str(cache_path),)) as pool:
            run_jobs(pool)
    else:
        run_jobs(None)

# ==================== 요약 파일 수집 ====================
def collect_summaries(base_dir):
    root = Path(base_dir)
    if not root.is_dir(): raise FileNotFoundError(root)
    records = []
    for model in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "Height_SUMMARY"):
        for mode in ("model_final", "model_mini"):
            path = model/mode/"repeat_test"/"height_summary.txt"
            if path.is_file():
                records.append(dict(model=model.name, mode=mode, text=path.read_text(encoding="utf-8")))
    return records

def ssh_args(source):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
            "-p", str(source.get("port", 22)), source["host"]]

def remote_run(source, script):
    command = [source.get("python", "python3"), "-", "--remote", source["base_dir"],
               source["gt_root"], str(source.get("workers", WORKERS)), str(int(FORCE))]
    subprocess.run(ssh_args(source) + [" ".join(shlex.quote(v) for v in command)],
                   input=script, text=True, check=True)

def remote_collect(source):
    script = "import json, sys\nfrom pathlib import Path\n" + inspect.getsource(collect_summaries) + \
             "\nprint(json.dumps(collect_summaries(sys.argv[1])))\n"
    command = [source.get("python", "python3"), "-", source["base_dir"]]
    result = subprocess.run(ssh_args(source) + [" ".join(shlex.quote(v) for v in command)],
                            input=script, text=True, capture_output=True, timeout=120, check=True)
    return json.loads(result.stdout)

# ==================== 로컬 Height_SUMMARY 수집 ====================
def save_collected(root, collected, errors):
    rows = []

    for server, records in collected.items():
        for r in records:
            output = root/server/r["model"]/r["mode"]/"height_summary.txt"
            output.parent.mkdir(parents=True, exist_ok=True)
            write_text(output, r["text"].splitlines())
            value = summary_values(r["text"])
            metrics = f"{value[0]:8d} {value[1]:12.6f} {value[2]:12.6f}" if value else f"{'-':>8} {'-':>12} {'-':>12}"
            rows.append(f"{server:<8} {r['model']:<75} {r['mode']:<12} {metrics}")

    lines = [
        "HEIGHT COMPARISON | unit: 10^-2 um | N: complete-10 bumps",
        "Model cohorts can differ. Invalid/version-mismatched results are shown as -.",
        f"{'Server':<8} {'Model':<75} {'CP':<12} {'N':>8} {'E_b':>12} {'sigma_b':>12}",
        "-"*132
    ] + rows

    if errors: lines += ["", "[ERRORS / CHECK SOURCE]"] + errors
    root.mkdir(parents=True, exist_ok=True)
    write_text(root/"Comparison.txt", lines)

# ==================== 실행 ====================
def main():
    global MODEL_ROOT, GT_ROOT, WORKERS, FORCE

    if len(sys.argv) > 1 and sys.argv[1] == "--remote":
        MODEL_ROOT, GT_ROOT = Path(sys.argv[2]), Path(sys.argv[3])
        WORKERS, FORCE = int(sys.argv[4]), bool(int(sys.argv[5]))
        run_local()
        return

    script = Path(__file__).read_text(encoding="utf-8")
    root, collected, errors = MODEL_ROOT/"Height_SUMMARY", {}, []
    sources = [dict(name="local", host="", base_dir=str(MODEL_ROOT))]
    sources += [s for s in REMOTE_SERVERS if s["host"].strip()]

    for source in sources:
        print(f"\n===== {source['name']}: HEIGHT EVALUATION =====", flush=True)

        try:
            if source["host"]: remote_run(source, script)
            else: run_local()
        except Exception as exc:
            errors.append(f"{source['name']} execution: {exc}")
            print(errors[-1], flush=True)

        try:
            collected[source["name"]] = remote_collect(source) if source["host"] else collect_summaries(source["base_dir"])
        except Exception as exc:
            message = getattr(exc, "stderr", None) or str(exc)
            errors.append(f"{source['name']} collection: {message.strip()}")
            print(errors[-1], flush=True)

        save_collected(root, collected, errors)

    print(f"\nSaved: {root/'Comparison.txt'}", flush=True)

if __name__ == "__main__":
    main()
