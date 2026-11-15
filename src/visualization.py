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
    (0.60, 0.60, 0.00),  # 12 right adrenal (olive; pale yellow was hard to read)
    (1.00, 0.00, 1.00),  # 13 left adrenal (magenta; light orange was indistinguishable from liver)
]

SHORT_NAMES = {
    1: "Spl", 2: "RKid", 3: "LKid", 4: "GB", 5: "Eso", 6: "Liv", 7: "Sto",
    8: "Ao", 9: "IVC", 10: "PSV", 11: "Pan", 12: "RAG", 13: "LAG",
}

_MUTED = "#6b7280"
_NAN_COLOR = "#e9ebf0"


def plot_heatmap(ax, mat, xlabels, ylabels, cmap="RdYlGn", vmin=0.0, vmax=1.0,
                 annotate_min=None, fmt="{:.2f}", title=None, subtitle=None, fontsize=8,
                 xrotation=0):
    """Annotated heatmap with white cell gutters, muted ticks and a left-aligned title.

    NaN cells are drawn in light grey. Values below ``annotate_min`` are not annotated.
    Returns the image handle (for ``heatmap_colorbar``).
    """
    mat = np.asarray(mat, dtype=float)
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(_NAN_COLOR)
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # white gutters between cells
    ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, colors=_MUTED, labelsize=fontsize + 1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(mat.shape[1]), xlabels, rotation=xrotation,
                  ha="right" if 0 < xrotation < 90 else "center",
                  rotation_mode="anchor" if 0 < xrotation < 90 else "default")
    ax.set_yticks(range(mat.shape[0]), ylabels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v) or (annotate_min is not None and v < annotate_min):
                continue
            r, g, b, _ = cmap(np.clip((v - vmin) / (vmax - vmin), 0, 1))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                    color="white" if lum < 0.5 else "#1f2937")

    # bold title, with an optional muted subtitle tucked between it and the axes
    if title:
        ax.set_title(title, loc="left", fontsize=fontsize + 5, fontweight="bold",
                     color="#111827", pad=(fontsize + 9) if subtitle else 6)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=fontsize + 1, color=_MUTED)
    return im


def heatmap_colorbar(fig, im, axes, label, extend="neither", shrink=0.9):
    """Slim shared colorbar with muted styling for a row of ``plot_heatmap`` axes."""
    cb = fig.colorbar(im, ax=axes, shrink=shrink, aspect=30, pad=0.015, extend=extend)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=_MUTED, labelsize=8)
    cb.set_label(label, color=_MUTED)
    return cb


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
def _overlay(ax, ct2d, lab2d, alpha=1.0, title=None):
    """Solid label colours on CT (the raw CT is shown separately), so they match the legend."""
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
            d = "\n" + "\n".join(f"{BTCV_ORGANS[o]}: Dice {dice(p == o, gt == o):.2f}"
                                 for o in organs)
        else:
            d = ""
        _overlay(ax, cut(c2), cut(sel(sl(p, z))), title=f"{m.capitalize()}{d}")
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


# ------------------------------------------------------- report (print) figures
ERROR_COLORS = {"False positive": (1.0, 0.15, 0.15), "False negative": (0.15, 0.45, 1.0),
                "Wrong organ": (0.1, 0.9, 0.2)}


def body_box(ct2d_hu, threshold=-500, margin=4):
    """Bounding box (r0, r1, c0, c1) of the patient's body (largest component above `threshold` HU).

    Drops the surrounding air and the scanner table, which otherwise fill most of each panel.
    """
    lab, n = ndimage.label(ndimage.binary_opening(ct2d_hu > threshold, iterations=2))
    if n == 0:
        return 0, ct2d_hu.shape[0], 0, ct2d_hu.shape[1]
    body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return crop_box(body, margin=margin)


def plot_case_figure(ct, gt, pred, spacing, rows, width_in=6.5, fontsize=8, legend_organs=None):
    """Compact CT | GT | prediction | error grid sized for a printed page (no wasted margins).

    rows: list of dicts with keys
        label  : row label (left margin)
        index  : slice index
        view   : "axial" (default) or "coronal"
        crop   : (r0, r1, c0, c1) in the displayed slice; default = body bounding box
    Panels are drawn in physical (mm) proportions; each row's height follows its crop so every
    panel spans the full column width. Column titles appear only once, on top.
    """
    prepared, heights = [], []
    for r in rows:
        view = r.get("view", "axial")
        sl = axial if view == "axial" else coronal
        idx = r["index"]
        hu2, g2, p2 = sl(ct, idx), sl(gt, idx), sl(pred, idx)
        r0, r1, c0, c1 = r.get("crop") or body_box(hu2)
        cut = lambda a: a[r0:r1, c0:c1]  # noqa: E731
        # displayed (rows, cols) pixel size in mm: axial = (y, x), coronal = (z, x)
        px = (spacing[1], spacing[0]) if view == "axial" else (spacing[2], spacing[0])
        h_mm, w_mm = (r1 - r0) * px[0], (c1 - c0) * px[1]
        prepared.append((r["label"], cut(window_ct(hu2)), cut(g2), cut(p2), (w_mm, h_mm)))
        heights.append(h_mm / w_mm)

    left, right, top, bottom, gap = 0.30, 0.02, 0.20, 0.55, 0.03  # inches
    panel_w = (width_in - left - right - 3 * gap) / 4
    panel_h = [panel_w * h for h in heights]
    height_in = top + bottom + sum(panel_h) + gap * (len(rows) - 1)
    fig = plt.figure(figsize=(width_in, height_in))

    y = height_in - top
    organs_seen = set()
    for i, ((label, c2, g2, p2, (w_mm, h_mm)), ph) in enumerate(zip(prepared, panel_h)):
        y -= ph
        ext = (0, w_mm, h_mm, 0)
        axes = [fig.add_axes(((left + j * (panel_w + gap)) / width_in, y / height_in,
                              panel_w / width_in, ph / height_in)) for j in range(4)]
        axes[0].imshow(c2, cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=ext)
        _overlay(axes[1], c2, g2)
        _overlay(axes[2], c2, p2)
        _error_map(axes[3], c2, g2, p2)
        for a in axes:
            for im in a.get_images():
                im.set_extent(ext)
            a.set_xlim(0, w_mm), a.set_ylim(h_mm, 0)
            a.set_axis_off()
        if i == 0:
            for a, t in zip(axes, ("CT", "Ground truth", "Prediction", "Error map")):
                a.set_title(t, fontsize=fontsize + 1, pad=3)
        axes[0].text(-0.03, 0.5, label, transform=axes[0].transAxes, rotation=90,
                     va="center", ha="right", fontsize=fontsize)
        organs_seen |= set(np.unique(g2)) | set(np.unique(p2))
        y -= gap

    organs = legend_organs or sorted(int(o) for o in organs_seen if o != 0)
    handles = [Patch(color=ORGAN_COLORS[o], label=BTCV_ORGANS[o]) for o in organs]
    handles += [Patch(color=c, label=n) for n, c in ERROR_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=6,
               fontsize=fontsize - 1, frameon=False, handlelength=1.0, handleheight=0.8,
               columnspacing=0.9, handletextpad=0.4, labelspacing=0.3)
    return fig


def plot_case_2x2(ct, gt, pred, spacing, index, view="axial", crop=None, width_in=3.25,
                  fontsize=7, legend_ncol=3, legend_min_rows=0):
    """One slice as a 2x2 grid (CT, error map / ground truth, prediction) for a half-width figure.

    legend_min_rows reserves space for at least that many legend rows, so figures placed side by
    side come out the same height even when their legends differ in length.
    """
    sl = axial if view == "axial" else coronal
    hu2, g2, p2 = sl(ct, index), sl(gt, index), sl(pred, index)
    r0, r1, c0, c1 = crop or body_box(hu2)
    c2, g2, p2 = (a[r0:r1, c0:c1] for a in (window_ct(hu2), g2, p2))
    px = (spacing[1], spacing[0]) if view == "axial" else (spacing[2], spacing[0])
    h_mm, w_mm = (r1 - r0) * px[0], (c1 - c0) * px[1]

    organs = sorted(int(o) for o in set(np.unique(g2)) | set(np.unique(p2)) if o != 0)
    handles = [Patch(color=ORGAN_COLORS[o], label=BTCV_ORGANS[o]) for o in organs]
    handles += [Patch(color=c, label=n) for n, c in ERROR_COLORS.items()]
    legend_rows = max(-(-len(handles) // legend_ncol), legend_min_rows)

    side, top, gap, title_h = 0.02, 0.17, 0.04, 0.16  # inches
    bottom = 0.08 + legend_rows * fontsize * 1.45 / 72
    panel_w = (width_in - 2 * side - gap) / 2
    panel_h = panel_w * h_mm / w_mm
    height_in = top + bottom + 2 * panel_h + gap + title_h
    fig = plt.figure(figsize=(width_in, height_in))

    ext = (0, w_mm, h_mm, 0)
    panels = ("CT", "Error map", "Ground truth", "Prediction")  # row-major: top row, then bottom
    for k, t in enumerate(panels):
        row, col = divmod(k, 2)
        y = height_in - top - (row + 1) * panel_h - row * (gap + title_h)
        ax = fig.add_axes(((side + col * (panel_w + gap)) / width_in, y / height_in,
                           panel_w / width_in, panel_h / height_in))
        if t == "CT":
            ax.imshow(c2, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        elif t == "Error map":
            _error_map(ax, c2, g2, p2)
        else:
            _overlay(ax, c2, g2 if t == "Ground truth" else p2)
        for im in ax.get_images():
            im.set_extent(ext)
        ax.set_xlim(0, w_mm), ax.set_ylim(h_mm, 0)
        ax.set_axis_off()
        ax.set_title(t, fontsize=fontsize + 1, pad=2)

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, (bottom - 0.03) / height_in), ncol=legend_ncol,
               fontsize=fontsize - 0.5, frameon=False, handlelength=1.0, handleheight=0.7,
               columnspacing=0.8, handletextpad=0.35, labelspacing=0.25, borderaxespad=0.2)
    return fig


def prepare_slice(ct, gt, pred, spacing, index, view="axial", crop=None):
    """Windowed CT, GT and prediction for one slice, cropped (default: body box), plus pixel size in mm."""
    sl = axial if view == "axial" else coronal
    hu2, g2, p2 = sl(ct, index), sl(gt, index), sl(pred, index) if pred is not None else None
    r0, r1, c0, c1 = crop or body_box(hu2)
    cut = lambda a: None if a is None else a[r0:r1, c0:c1]  # noqa: E731
    px = (spacing[1], spacing[0]) if view == "axial" else (spacing[2], spacing[0])
    return cut(window_ct(hu2)), cut(g2), cut(p2), px


def plot_panel_grid(rows, col_titles, width_in=6.5, fontsize=8, legend_organs=None,
                    error_legend=True, legend_ncol=6, gap=0.03):
    """Generic print-sized grid of slice panels with column titles on top and row labels on the left.

    rows: list of dicts
        label  : row label (optional)
        ct     : windowed, cropped 2-D CT (values 0..1)
        px     : (row, col) pixel size in mm, so panels keep physical proportions
        panels : one entry per column, each (kind, data, note) with
                 kind "ct" (data ignored), "labels" (data = label slice) or "errors" (data = (gt, pred));
                 note = optional short text drawn in the panel's lower-left corner
    The figure height follows the crops, so every panel spans its full column width.
    """
    ncols = len(col_titles)
    n_label_lines = max((r["label"].count("\n") + 1 for r in rows if r.get("label")), default=0)
    left = 0.04 + 0.14 * n_label_lines
    right, top = 0.02, 0.17
    seen = set()
    for r in rows:
        for kind, data, _ in r["panels"]:
            if kind == "labels":
                seen |= set(np.unique(data))
            elif kind == "errors":
                seen |= set(np.unique(data[0])) | set(np.unique(data[1]))
    organs = legend_organs if legend_organs is not None else sorted(int(o) for o in seen if o != 0)
    handles = [Patch(color=ORGAN_COLORS[o], label=BTCV_ORGANS[o]) for o in organs]
    if error_legend:
        handles += [Patch(color=c, label=n) for n, c in ERROR_COLORS.items()]
    n_leg = -(-len(handles) // legend_ncol)
    bottom = 0.06 + n_leg * fontsize * 1.45 / 72

    panel_w = (width_in - left - right - (ncols - 1) * gap) / ncols
    sizes = [(r["ct"].shape[0] * r["px"][0], r["ct"].shape[1] * r["px"][1]) for r in rows]
    panel_h = [panel_w * h / w for h, w in sizes]
    height_in = top + bottom + sum(panel_h) + gap * (len(rows) - 1)
    fig = plt.figure(figsize=(width_in, height_in))

    y = height_in - top
    for i, (r, ph, (h_mm, w_mm)) in enumerate(zip(rows, panel_h, sizes)):
        y -= ph
        ext = (0, w_mm, h_mm, 0)
        for j, (kind, data, note) in enumerate(r["panels"]):
            ax = fig.add_axes(((left + j * (panel_w + gap)) / width_in, y / height_in,
                               panel_w / width_in, ph / height_in))
            if kind == "ct":
                ax.imshow(r["ct"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            elif kind == "labels":
                _overlay(ax, r["ct"], data)
            else:
                _error_map(ax, r["ct"], *data)
            for im in ax.get_images():
                im.set_extent(ext)
            ax.set_xlim(0, w_mm), ax.set_ylim(h_mm, 0)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(col_titles[j], fontsize=fontsize + 1, pad=3)
            if note:
                ax.text(0.03, 0.04, note, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=fontsize - 1, color="white", linespacing=1.15,
                        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=1.5))
            if j == 0 and r.get("label"):
                ax.text(-0.03, 0.5, r["label"], transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=fontsize)
        y -= gap

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, (bottom - 0.02) / height_in),
               ncol=legend_ncol, fontsize=fontsize - 0.5, frameon=False, handlelength=1.0,
               handleheight=0.7, columnspacing=0.9, handletextpad=0.35, labelspacing=0.25,
               borderaxespad=0.2)
    return fig


def error_row(ct, gt, pred, spacing, index, label=None, view="axial", crop=None):
    """Row for plot_panel_grid in the report order: CT | error map | ground truth | prediction."""
    c2, g2, p2, px = prepare_slice(ct, gt, pred, spacing, index, view, crop)
    return dict(label=label, ct=c2, px=px, panels=[("ct", None, None), ("errors", (g2, p2), None),
                                                   ("labels", g2, None), ("labels", p2, None)])


ERROR_COLUMNS = ("CT", "Error map", "Ground truth", "Prediction")


def expand_to_aspect(box, aspect, shape):
    """Grow box (r0, r1, c0, c1) symmetrically so height/width == aspect, staying inside `shape`."""
    r0, r1, c0, c1 = box
    h, w = r1 - r0, c1 - c0
    if h / w < aspect:
        h = int(round(w * aspect))
    else:
        w = int(round(h / aspect))
    rc, cc = (r0 + r1) / 2, (c0 + c1) / 2
    r0 = int(min(max(rc - h / 2, 0), shape[0] - h)); c0 = int(min(max(cc - w / 2, 0), shape[1] - w))
    return r0, r0 + h, c0, c0 + w


def block_2x2(ct, gt, pred, spacing, index, title=None, view="axial", crop=None):
    """Block for plot_blocks_2x2 in the report order: CT, error map / ground truth, prediction."""
    c2, g2, p2, px = prepare_slice(ct, gt, pred, spacing, index, view, crop)
    return dict(title=title, ct=c2, px=px, panels=[
        ("ct", None, None, "CT"), ("errors", (g2, p2), None, "Error map"),
        ("labels", g2, None, "Ground truth"), ("labels", p2, None, "Prediction")])


def plot_blocks_2x2(blocks, block_width_in=3.25, block_gap_in=0.15, fontsize=7, legend_organs=None,
                    error_legend=True, legend_ncol=None):
    """One or more 2x2 blocks side by side (same style as plot_case_2x2) with a shared legend.

    blocks: list of dicts with title (optional, above the block), ct, px and four panels
            (kind, data, note, panel_title), kind as in plot_panel_grid.
    """
    side, gap = 0.02, 0.04
    title_h = 0.16  # space for each panel title
    block_title_h = 0.17 if any(b.get("title") for b in blocks) else 0.0
    seen = set()
    for b in blocks:
        for kind, data, _, _ in b["panels"]:
            if kind == "labels":
                seen |= set(np.unique(data))
            elif kind == "errors":
                seen |= set(np.unique(data[0])) | set(np.unique(data[1]))
    organs = legend_organs if legend_organs is not None else sorted(int(o) for o in seen if o != 0)
    handles = [Patch(color=ORGAN_COLORS[o], label=BTCV_ORGANS[o]) for o in organs]
    if error_legend:
        handles += [Patch(color=c, label=n) for n, c in ERROR_COLORS.items()]
    ncol = legend_ncol or 3 * len(blocks)
    n_leg = -(-len(handles) // ncol)
    bottom = 0.06 + n_leg * fontsize * 1.45 / 72

    width_in = len(blocks) * block_width_in + (len(blocks) - 1) * block_gap_in
    panel_w = (block_width_in - 2 * side - gap) / 2
    sizes = [(b["ct"].shape[0] * b["px"][0], b["ct"].shape[1] * b["px"][1]) for b in blocks]
    panel_h = [panel_w * h / w for h, w in sizes]
    height_in = block_title_h + 2 * title_h + 2 * max(panel_h) + gap + bottom
    fig = plt.figure(figsize=(width_in, height_in))

    for bi, (b, ph, (h_mm, w_mm)) in enumerate(zip(blocks, panel_h, sizes)):
        x0 = bi * (block_width_in + block_gap_in)
        if b.get("title"):
            fig.text((x0 + block_width_in / 2) / width_in, 1 - 0.02 / height_in, b["title"],
                     ha="center", va="top", fontsize=fontsize + 1.5, fontweight="bold")
        ext = (0, w_mm, h_mm, 0)
        for k, (kind, data, note, ptitle) in enumerate(b["panels"]):
            row, col = divmod(k, 2)
            y = height_in - block_title_h - title_h - row * (ph + gap + title_h) - ph
            ax = fig.add_axes(((x0 + side + col * (panel_w + gap)) / width_in, y / height_in,
                               panel_w / width_in, ph / height_in))
            if kind == "ct":
                ax.imshow(b["ct"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            elif kind == "labels":
                _overlay(ax, b["ct"], data)
            else:
                _error_map(ax, b["ct"], *data)
            for im in ax.get_images():
                im.set_extent(ext)
            ax.set_xlim(0, w_mm), ax.set_ylim(h_mm, 0)
            ax.set_axis_off()
            ax.set_title(ptitle, fontsize=fontsize + 1, pad=2)
            if note:
                ax.text(0.03, 0.04, note, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=fontsize - 0.5, color="white",
                        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=1.5))

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, (bottom - 0.02) / height_in),
               ncol=ncol, fontsize=fontsize - 0.5, frameon=False, handlelength=1.0,
               handleheight=0.7, columnspacing=0.8, handletextpad=0.35, labelspacing=0.25,
               borderaxespad=0.2)
    return fig
