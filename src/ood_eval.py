"""Bonus task (OOD): score Swin UNETR predictions on the TotalSegmentator (dataset v300) test split.

Steps: `src/ood_prepare.py` (labels, survey, case lists) -> `src/ood_infer.py` (GPU) -> this script ->
`notebooks/05_ood_eval.ipynb` (tables and figures).

The GPU step saves argmax predictions in the model's working grid
(1.5 x 1.5 x 2.0 mm, RAS). This script does the CPU part, with the same functions as Tasks 2 and 3:
  1. resample each prediction back to the label grid with the official `resample_3d` (nearest neighbour)
  2. Dice per organ (official definition) and HD95 / HD in mm (Task 3 implementation)

Ground truth: BTCV-indexed label maps merged from the TotalSegmentator masks (`ts_btcv/<id>_label.nii.gz`).
Usage: python -m src.ood_eval [--models base small tiny] [--workers 6]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from src.config import BTCV_ORGANS, METRICS_DIR, OOD_LABEL_DIR, OOD_PRED_RAW_DIR, PREDICTIONS_DIR
from src.evaluate import hausdorff_distances
from src.inference import compute_dice, resample_3d

LABEL_DIR = OOD_LABEL_DIR
PRED_RAW_DIR = OOD_PRED_RAW_DIR  # from src/ood_infer.py (GPU), in the 1.5 x 1.5 x 2.0 mm model grid
PRED_NATIVE_DIR = PREDICTIONS_DIR / "ood"  # resampled to the label grid (written by this script)


def evaluate_case(args: tuple[str, str]) -> list[dict]:
    model_key, case = args
    label_img = nib.load(LABEL_DIR / f"{case}_label.nii.gz")
    pred_img = nib.load(PRED_RAW_DIR / model_key / f"{case}.nii.gz")
    # Predictions are in RAS; the TotalSegmentator test labels are stored in RAS too (checked in the survey).
    # Fail loudly rather than silently compare misaligned volumes.
    for name, img in (("label", label_img), ("pred", pred_img)):
        if nib.aff2axcodes(img.affine) != ("R", "A", "S"):
            raise ValueError(f"{case}: {name} is {nib.aff2axcodes(img.affine)}, expected RAS")

    gt = np.asarray(label_img.dataobj).astype(np.uint8)
    pred = resample_3d(np.asarray(pred_img.dataobj).astype(np.uint8), gt.shape)
    spacing = tuple(float(s) for s in label_img.header.get_zooms()[:3])

    out = PRED_NATIVE_DIR / model_key / f"{case}.nii.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred, label_img.affine), str(out))

    rows = []
    for idx, organ in BTCV_ORGANS.items():
        p, g = pred == idx, gt == idx
        hd95, hd = hausdorff_distances(p, g, spacing)
        rows.append(
            dict(case=case, organ=organ, gt_voxels=int(g.sum()), pred_voxels=int(p.sum()),
                 dice=compute_dice(p, g), hd95=hd95, hd=hd)
        )
    return rows


def organ_geometry(case: str) -> list[dict]:
    """Per organ: is it cut off by the scan's field of view? (GT touches the first or last axial slice)."""
    label_img = nib.load(LABEL_DIR / f"{case}_label.nii.gz")
    gt = np.asarray(label_img.dataobj).astype(np.uint8)
    z_mm = gt.shape[2] * float(label_img.header.get_zooms()[2])
    rows = []
    for idx, organ in BTCV_ORGANS.items():
        g = gt == idx
        rows.append(dict(case=case, organ=organ, scan_z_mm=z_mm,
                         truncated=bool(g.any() and (g[:, :, 0].any() or g[:, :, -1].any()))))
    return rows


def largest_cc_hd95(args: tuple[str, str]) -> list[dict]:
    """HD95 of the full prediction vs. its largest connected component only (26-connectivity, as in Task 3).
    A large drop means distant fragments, not the organ boundary, drive HD95."""
    model_key, case = args
    label_img = nib.load(LABEL_DIR / f"{case}_label.nii.gz")
    gt = np.asarray(label_img.dataobj).astype(np.uint8)
    pred = np.asarray(nib.load(PRED_NATIVE_DIR / model_key / f"{case}.nii.gz").dataobj).astype(np.uint8)
    spacing = tuple(float(s) for s in label_img.header.get_zooms()[:3])
    rows = []
    for idx, organ in BTCV_ORGANS.items():
        p, g = pred == idx, gt == idx
        if not (p.any() and g.any()):
            continue
        lab, n = ndimage.label(p, structure=np.ones((3, 3, 3)))
        main = lab == np.bincount(lab.ravel())[1:].argmax() + 1
        rows.append(dict(model=model_key, case=case, organ=organ, components=int(n),
                         fragment_voxels=int((p & ~main).sum()),
                         hd95_largest_cc=hausdorff_distances(main, g, spacing)[0]))
    return rows


def compute_extras(models: list[str], workers: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Geometry (truncation) table and largest-component HD95 table, cached as CSVs."""
    cases = sorted(p.name.removesuffix("_label.nii.gz") for p in LABEL_DIR.glob("*_label.nii.gz")
                   if (PRED_NATIVE_DIR / models[0] / p.name.replace("_label", "")).exists())
    geo_csv, cc_csv = METRICS_DIR / "ood_totalseg_geometry.csv", METRICS_DIR / "ood_totalseg_largest_cc.csv"
    with ProcessPoolExecutor(workers) as ex:
        if not geo_csv.exists():
            pd.DataFrame([r for rs in ex.map(organ_geometry, cases) for r in rs]).to_csv(geo_csv, index=False)
        if not cc_csv.exists():
            jobs = [(m, c) for m in models for c in cases]
            pd.DataFrame([r for rs in ex.map(largest_cc_hd95, jobs) for r in rs]).to_csv(cc_csv, index=False)
    return pd.read_csv(geo_csv), pd.read_csv(cc_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "small", "tiny"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    for model_key in a.models:
        out = METRICS_DIR / f"ood_totalseg_{model_key}_overlap0.5.csv"
        old = pd.read_csv(out) if out.exists() else pd.DataFrame()
        done = set(old.case) if len(old) else set()
        cases = sorted(p.name.removesuffix(".nii.gz") for p in (PRED_RAW_DIR / model_key).glob("*.nii.gz"))
        new = [c for c in cases if c not in done]  # incremental: only score cases not in the CSV yet
        if new:
            with ProcessPoolExecutor(a.workers) as ex:
                rows = [r for case_rows in ex.map(evaluate_case, [(model_key, c) for c in new]) for r in case_rows]
            df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).sort_values(["case", "organ"])
            df.to_csv(out, index=False)
        else:
            df = old
        present = df[df.gt_voxels > 0] if len(df) else df
        print(f"[{model_key}] +{len(new)} new, {df.case.nunique() if len(df) else 0} cases total -> {out.name}; "
              f"mean Dice over present organs = {present.dice.mean() if len(df) else float('nan'):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
