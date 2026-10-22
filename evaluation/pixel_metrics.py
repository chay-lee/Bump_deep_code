from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image


SHAPE, SUBSTRATE_PERCENTILE = (232, 232), 10


def fixed_substrate_mask(path):
    gt = np.loadtxt(path, delimiter=",")
    if gt.ndim != 2 or not np.isfinite(gt).all(): raise ValueError("INVALID_MASK_GT")
    if gt.shape != SHAPE:
        gt = np.asarray(Image.fromarray(gt.astype(np.float32)).resize(SHAPE[::-1], Image.Resampling.BILINEAR))
    if np.ptp(gt) <= 1e-8: raise ValueError("GT_NO_CONTRAST")
    return gt <= np.percentile(gt, SUBSTRATE_PERCENTILE)


class PixelEvaluator:
    def __init__(self, scans, reference_dir, output_dir):
        self.scans, self.reference_dir, self.output_dir = tuple(scans), Path(reference_dir), Path(output_dir)
        self.states, self.errors, self.seen = {}, Counter(), set()

    def update(self, scan, stem, pred, gt):
        if scan not in self.scans or (scan, stem) in self.seen:
            raise ValueError(f"Duplicate/invalid sample: {scan}/{stem}")
        self.seen.add((scan, stem)); pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
        if pred.shape != SHAPE or not np.isfinite(pred).all():
            self.errors["INVALID_PRED"] += 1; return

        if stem not in self.states:
            try: mask = fixed_substrate_mask(self.reference_dir/f"{stem}.csv")
            except (OSError, ValueError) as exc:
                mask = None; self.errors[f"MASK: {exc}"] += 1
            self.states[stem] = [0, np.zeros(SHAPE), np.zeros(SHAPE), {}, mask]

        state = self.states[stem]; state[0] += 1
        delta = pred-state[1]; state[1] += delta/state[0]; state[2] += delta*(pred-state[1])
        if gt.shape == SHAPE and np.isfinite(gt).all(): state[3][scan] = float(np.abs(pred-gt).mean())
        else: self.errors["INVALID_OR_MISSING_PIXEL_GT"] += 1

    def finish(self):
        complete = {k: v for k, v in self.states.items() if v[0] == len(self.scans)}
        paired = {k: v for k, v in complete.items() if len(v[3]) == len(self.scans)}
        substrate = {k: v for k, v in complete.items() if v[4] is not None}
        maes = [float(np.mean([v[3][s] for v in paired.values()])) if paired else np.nan for s in self.scans]
        stds = {k: np.sqrt(np.maximum(v[2]/v[0], 0)) for k, v in complete.items()} if len(self.scans) > 1 else {}

        ep = float(np.mean(maes))
        sp = float(np.mean([v.mean() for v in stds.values()])) if stds else np.nan
        ss = float(np.mean([stds[k][v[4]].mean() for k, v in substrate.items()])) if stds and substrate else np.nan

        lines = ["PIXEL METRICS", "Unit: 10^-2 um | population STD (ddof=0)",
                 f"Scans: {list(self.scans)} | Seen samples: {len(self.seen)}",
                 f"Complete prediction ROIs: {len(complete)} | Paired MAE ROIs: {len(paired)} | Substrate ROIs: {len(substrate)}",
                 f"Incomplete prediction ROIs: {len(self.states)-len(complete)} | Invalid predictions: {self.errors['INVALID_PRED']}",
                 f"Fixed substrate mask: {self.reference_dir} / GT <= P{SUBSTRATE_PERCENTILE}; Pillow bilinear if needed.",
                 "No offset subtraction or plane correction. Each metric uses its listed complete-scan cohort.",
                 f"E_p: {100*ep:.6f}", f"sigma_p: {100*sp:.6f}", f"sigma_s: {100*ss:.6f}",
                 "", f"{'Scan':>6} {'N':>8} {'MAE':>14}"]
        lines += [f"{s:6} {len(paired):8} {100*v:14.6f}" for s, v in zip(self.scans, maes)]
        lines += ["", "Repeatability requires at least two scans.", f"Errors: {dict(self.errors)}"]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir/"pixel_metrics.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
        with (self.output_dir/"pixel_cohort.csv").open("w", encoding="utf-8") as f:
            f.write("stem,pred_complete,mae_complete,substrate_complete\n")
            for stem in sorted({s for _, s in self.seen}):
                f.write(f"{stem},{int(stem in complete)},{int(stem in paired)},{int(stem in substrate)}\n")

        self.states.clear()
        return {"E_p": ep, "sigma_p": sp, "sigma_s": ss}