"""
Task 3: Quantitative evaluation of saved Swin UNETR predictions.

Loads the predicted masks written by `run_btcv_inference` (results/predictions/<model>_overlap<x>/)
and the BTCV ground-truth labels, then computes per organ:
- Dice
- HD95 and HD (max) in millimetres, using the label's voxel spacing
"""

import json

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from src.config import (
    BTCV_ORGANS,
    DATA_DIR,
    DATASET_JSON_PATH,
    METRICS_DIR,
    PREDICTIONS_DIR,
)
from src.inference import compute_dice


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels of a binary mask (voxels removed by one erosion step)."""
    return mask & ~ndimage.binary_erosion(mask)


def hausdorff_distances(
    pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[float, float]:
    """
    Return (HD95, HD) in mm between two binary masks.

    HD95 follows the MONAI convention: the 95th percentile of surface distances is taken in each
    direction (pred->gt and gt->pred) and the larger of the two is reported.
    Returns (nan, nan) if either mask is empty, because the distance is undefined.
    """
    if not pred.any() or not gt.any():
        return float("nan"), float("nan")

    # Crop to the joint bounding box (plus margin) to keep the distance transform fast
    coords = np.argwhere(pred | gt)
    lo = np.maximum(coords.min(axis=0) - 2, 0)
    hi = coords.max(axis=0) + 3
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    pred, gt = pred[box], gt[box]

    pred_surf, gt_surf = _surface(pred), _surface(gt)
    # Distance (mm) from every voxel to the nearest surface voxel of the other mask
    dist_to_gt = ndimage.distance_transform_edt(~gt_surf, sampling=spacing)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surf, sampling=spacing)

    d_pred_to_gt = dist_to_gt[pred_surf]
    d_gt_to_pred = dist_to_pred[gt_surf]

    hd95 = max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))
    hd = max(d_pred_to_gt.max(), d_gt_to_pred.max())

    return float(hd95), float(hd)


def evaluate_model(model_key: str, infer_overlap: float = 0.5, save: bool = True) -> pd.DataFrame:
    """
    Compute Dice, HD95 and HD for every validation case and organ of one model.
    Returns a long-format table: model, case, organ, dice, hd95, hd.
    """
    pred_dir = PREDICTIONS_DIR / f"{model_key}_overlap{infer_overlap}"
    with open(DATASET_JSON_PATH) as f:
        val_items = json.load(f)["validation"]

    rows = []
    for item in val_items:
        case = item["image"].split("/")[-1]
        pred_path = pred_dir / case
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction {pred_path}. Run the Task 2 notebook first.")

        label_img = nib.load(str(DATA_DIR / item["label"]))
        gt = np.asarray(label_img.dataobj).astype(np.uint8)
        pred = np.asarray(nib.load(str(pred_path)).dataobj).astype(np.uint8)
        assert pred.shape == gt.shape, f"{case}: prediction {pred.shape} vs label {gt.shape}"
        spacing = tuple(float(s) for s in label_img.header.get_zooms()[:3])

        for organ_idx, organ_name in BTCV_ORGANS.items():
            p, g = pred == organ_idx, gt == organ_idx
            hd95, hd = hausdorff_distances(p, g, spacing)
            rows.append(
                {
                    "model": model_key,
                    "case": case,
                    "organ": organ_name,
                    "dice": compute_dice(p, g),
                    "hd95": hd95,
                    "hd": hd,
                    "organ_in_gt": bool(g.any()),
                }
            )

    df = pd.DataFrame(rows)
    if save:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(METRICS_DIR / f"btcv_metrics_{model_key}_overlap{infer_overlap}.csv", index=False)

    return df
