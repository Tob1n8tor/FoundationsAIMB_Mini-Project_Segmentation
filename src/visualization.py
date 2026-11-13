"""Helpers for qualitative inspection of BTCV segmentations (Task 4)."""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage

from src.config import BTCV_ORGANS, DATA_DIR, PREDICTIONS_DIR, PREPROCESS_CONFIG

N_CLASSES = len(BTCV_ORGANS) + 1  # 13 organs + background

# Fixed, distinguishable colour per organ (index 0 = background, transparent in overlays)
ORGAN_COLORS = [
    (0.00, 0.00, 0.00),  # 0  background
    (0.89, 0.10, 0.11),  # 1  spleen
    (0.22, 0.49, 0.72),  # 2  right kidney
    (0.30, 0.69, 0.29),  # 3  left kidney
    (1.00, 0.85, 0.00),  # 4  gallbladder
    (0.60, 0.31, 0.64),  # 5  esophagus
    (1.00, 0.50, 0.00),  # 6  liver
    (0.65, 0.34, 0.16),  # 7  stomach
    (0.97, 0.51, 0.75),  # 8  aorta
    (0.00, 0.80, 0.80),  # 9  IVC
    (0.50, 0.50, 1.00),  # 10 portal & splenic veins
    (0.70, 0.87, 0.54),  # 11 pancreas
    (1.00, 1.00, 0.60),  # 12 right adrenal
    (0.99, 0.75, 0.44),  # 13 left adrenal
]

SHORT_NAMES = {
    1: "Spl", 2: "RKid", 3: "LKid", 4: "GB", 5: "Eso", 6: "Liv", 7: "Sto",
    8: "Ao", 9: "IVC", 10: "PSV", 11: "Pan", 12: "RAG", 13: "LAG",
}


def btcv_cmap() -> ListedColormap:
    return ListedColormap(ORGAN_COLORS, name="btcv")


def organ_legend_handles(organs=None):
    organs = organs if organs is not None else list(BTCV_ORGANS)
    return [Patch(color=ORGAN_COLORS[i], label=BTCV_ORGANS[i]) for i in organs]


# --------------------------------------------------------------------------- I/O
def case_paths(case_id: str, model_key: str = "base", overlap: float = 0.5):
    """case_id like '0035'."""
    return (
        DATA_DIR / "imagesTr" / f"img{case_id}.nii.gz",
        DATA_DIR / "labelsTr" / f"label{case_id}.nii.gz",
        PREDICTIONS_DIR / f"{model_key}_overlap{overlap}" / f"img{case_id}.nii.gz",
    )


def load_volume(path: Path, dtype=None) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).dataobj)
    return arr.astype(dtype) if dtype is not None else arr


def load_case(case_id: str, model_key: str = "base", overlap: float = 0.5):
    """Return (ct [HU], gt, pred, spacing) in the native voxel grid."""
    img_p, lab_p, pred_p = case_paths(case_id, model_key, overlap)
    img = nib.load(str(img_p))
    ct = np.asarray(img.dataobj).astype(np.float32)
    gt = load_volume(lab_p, np.uint8)
    pred = load_volume(pred_p, np.uint8)
    assert ct.shape == gt.shape == pred.shape, (ct.shape, gt.shape, pred.shape)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return ct, gt, pred, spacing


def window_ct(ct, a_min=PREPROCESS_CONFIG["a_min"], a_max=PREPROCESS_CONFIG["a_max"]):
    return np.clip((ct - a_min) / (a_max - a_min), 0, 1)


# ------------------------------------------------------------------ slicing
def axial(vol: np.ndarray, z: int) -> np.ndarray:
    """Axial slice in radiological convention (LAS data: anterior up, patient-right on viewer-left)."""
    return np.rot90(vol[:, :, z])


def coronal(vol: np.ndarray, y: int) -> np.ndarray:
    """Coronal slice (superior up, patient-right on viewer-left)."""
    return np.rot90(vol[:, y, :])


def pick_slice(gt, pred, organ=None, mode="max_gt", axis=2) -> int:
    """
    Choose a slice index along `axis`.
    mode='max_gt'    -> slice with the largest GT area of `organ` (or all foreground)
    mode='max_pred'  -> slice with the largest predicted area of `organ`
    mode='max_error' -> slice with the most FP+FN voxels for `organ` (or all foreground)
    """
    g = gt == organ if organ is not None else gt > 0
    p = pred == organ if organ is not None else pred > 0
    if mode == "max_gt":
        score = g
    elif mode == "max_pred":
        score = p
    elif mode == "max_error":
        score = g ^ p if organ is not None else (gt != pred)
    else:
        raise ValueError(mode)
    axes = tuple(a for a in range(3) if a != axis)
    return int(np.argmax(score.sum(axis=axes)))


def dice(pred_mask, gt_mask) -> float:
    s = pred_mask.sum() + gt_mask.sum()
    return float(2 * np.logical_and(pred_mask, gt_mask).sum() / s) if s else float("nan")


# ------------------------------------------------------------------ plotting
def _overlay(ax, ct2d, lab2d, alpha=0.45, title=None):
    ax.imshow(ct2d, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    masked = np.ma.masked_where(lab2d == 0, lab2d)
    ax.imshow(masked, cmap=btcv_cmap(), vmin=0, vmax=N_CLASSES - 1, alpha=alpha,
              interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)


def _error_map(ax, ct2d, gt2d, pred2d, organs=None, title=None):
    """Red = false positive, blue = false negative, green = wrong organ (both fg but different)."""
    if organs is not None:
        gt2d = np.where(np.isin(gt2d, organs), gt2d, 0)
        pred2d = np.where(np.isin(pred2d, organs), pred2d, 0)
    rgb = np.repeat(ct2d[..., None], 3, axis=2) * 0.8
    fp = (pred2d > 0) & (gt2d == 0)
    fn = (gt2d > 0) & (pred2d == 0)
    wrong = (gt2d > 0) & (pred2d > 0) & (gt2d != pred2d)
    rgb[fp] = (1.0, 0.15, 0.15)
    rgb[fn] = (0.15, 0.45, 1.0)
    rgb[wrong] = (0.1, 0.9, 0.2)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)


def crop_box(*masks2d, margin=20):
    """Bounding box (r0, r1, c0, c1) around the union of 2-D masks."""
    m = np.zeros_like(masks2d[0], dtype=bool)
    for x in masks2d:
        m |= x
    if not m.any():
        return 0, m.shape[0], 0, m.shape[1]
    r, c = np.where(m)
    return (max(r.min() - margin, 0), min(r.max() + margin, m.shape[0]),
            max(c.min() - margin, 0), min(c.max() + margin, m.shape[1]))


def plot_case_slice(ct, gt, pred, z, organs=None, view="axial", crop=None,
                    axes=None, row_label=None, aspect=1.0):
    """Four panels: CT | GT overlay | prediction overlay | error map."""
    sl = axial if view == "axial" else coronal
    c2, g2, p2 = sl(window_ct(ct), z), sl(gt, z), sl(pred, z)
    if organs is not None:
        g2 = np.where(np.isin(g2, organs), g2, 0)
        p2 = np.where(np.isin(p2, organs), p2, 0)
    if crop is not None:
        r0, r1, c0, c1 = crop
        c2, g2, p2 = c2[r0:r1, c0:c1], g2[r0:r1, c0:c1], p2[r0:r1, c0:c1]
    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(c2, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_axis_off()
    axes[0].set_title("CT", fontsize=10)
    _overlay(axes[1], c2, g2, title="Ground truth")
    _overlay(axes[2], c2, p2, title="Prediction")
    _error_map(axes[3], c2, g2, p2, title="Errors (FP red / FN blue / wrong organ green)")
    for a in axes:
        a.set_aspect(aspect)
    if row_label:
        axes[0].text(-0.04, 0.5, row_label, transform=axes[0].transAxes, rotation=90,
                     va="center", ha="right", fontsize=11, fontweight="bold")
    return axes


def plot_model_comparison(case_id, z, organs=None, models=("base", "small", "tiny"),
                          crop=None, view="axial", aspect=1.0, axes=None):
    """CT | GT | one prediction overlay per model, with per-organ Dice in the titles."""
    ct, gt, _, _ = load_case(case_id, models[0])
    preds = {m: load_case(case_id, m)[2] for m in models}
    sl = axial if view == "axial" else coronal
    c2 = sl(window_ct(ct), z)
    sel = (lambda a: np.where(np.isin(a, organs), a, 0)) if organs is not None else (lambda a: a)
    g2 = sel(sl(gt, z))
    if crop is not None:
        r0, r1, c0, c1 = crop
        cut = lambda a: a[r0:r1, c0:c1]  # noqa: E731
    else:
        cut = lambda a: a  # noqa: E731
    if axes is None:
        _, axes = plt.subplots(1, 2 + len(models), figsize=(4 * (2 + len(models)), 4.4))
    axes[0].imshow(cut(c2), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_axis_off()
    axes[0].set_title(f"img{case_id} CT", fontsize=10)
    _overlay(axes[1], cut(c2), cut(g2), title="Ground truth")
    for ax, m in zip(axes[2:], models):
        p = preds[m]
        if organs is not None:
            d = ", ".join(f"{SHORT_NAMES[o]} {dice(p == o, gt == o):.2f}" for o in organs)
        else:
            d = ""
        _overlay(ax, cut(c2), cut(sel(sl(p, z))), title=f"{m.capitalize()}  {d}")
    for a in axes:
        a.set_aspect(aspect)
    return axes


# ------------------------------------------------------------------ analysis
def confusion_voxels(gt, pred, n=N_CLASSES) -> np.ndarray:
    """n x n voxel confusion matrix; rows = GT class, cols = predicted class."""
    idx = gt.astype(np.int64).ravel() * n + pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def component_stats(mask: np.ndarray):
    """Number of 3-D connected components and fraction of voxels outside the largest one."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), float(1 - sizes.max() / sizes.sum())
