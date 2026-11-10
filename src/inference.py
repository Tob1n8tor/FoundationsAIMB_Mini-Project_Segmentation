import inspect
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from scipy import ndimage
from tqdm import tqdm

from src.config import (
    BTCV_ORGANS,
    CHECKPOINTS_DIR,
    MODEL_CONFIGS,
    PREDICTIONS_DIR,
    PREPROCESS_CONFIG,
)


def get_device(prefer_mps: bool = False) -> torch.device:
    """
    Select compute device.
    Note: PyTorch MPS (Apple Silicon) does not yet support all 3D volumetric ops (e.g. reflection/replication
    pad3d or large 3D convs in MONAI), so CUDA is preferred if available, with CPU fallback.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resample_3d(img: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    """
    Exact reproduction of `resample_3d` from MONAI SwinUNETR/BTCV/utils/utils.py.
    Uses nearest-neighbor interpolation (order=0) so integer organ class IDs (0..13) are preserved.
    """
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (
        float(tx) / float(imx),
        float(ty) / float(imy),
        float(tz) / float(imz),
    )
    return ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Exact reproduction of `dice(x, y)` from MONAI SwinUNETR/BTCV/utils/utils.py."""
    intersect = np.sum(pred_mask * gt_mask)
    y_sum = np.sum(gt_mask)
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(pred_mask)
    return float(2.0 * intersect / (x_sum + y_sum))


def load_swin_unetr(
    model_key: str, device: torch.device, checkpoint_dir: Path = CHECKPOINTS_DIR
) -> SwinUNETR:
    """Instantiate SwinUNETR for `base`, `small`, or `tiny` and load official pretrained weights."""
    cfg = MODEL_CONFIGS[model_key]
    ckpt_path = checkpoint_dir / cfg["filename"]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run download_pretrained_weights() first."
        )

    # Handle MONAI version difference: `img_size` was required in MONAI < 1.3 and deprecated/removed in MONAI >=1.3
    swin_kwargs = {
        "in_channels": 1,
        "out_channels": 14,
        "feature_size": cfg["feature_size"],
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "dropout_path_rate": 0.0,
        "use_checkpoint": False,
    }
    if "img_size" in inspect.signature(SwinUNETR.__init__).parameters:
        swin_kwargs["img_size"] = PREPROCESS_CONFIG["roi_size"]

    model = SwinUNETR(**swin_kwargs)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def run_btcv_inference(
    model_key: str,
    val_loader,
    device: torch.device,
    infer_overlap: float = PREPROCESS_CONFIG["infer_overlap"],
    sw_batch_size: int = PREPROCESS_CONFIG["sw_batch_size"],
    save_predictions: bool = True,
) -> dict:
    """
    Run sliding-window inference for a given model configuration ('base', 'small', 'tiny')
    across all BTCV validation scans, save predicted NIfTI masks, and return organ-wise Dice scores.
    """
    model = load_swin_unetr(model_key, device=device)
    out_dir = PREDICTIONS_DIR / f"{model_key}_overlap{infer_overlap}"
    if save_predictions:
        out_dir.mkdir(parents=True, exist_ok=True)

    case_results = []

    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc=f"Inference [{MODEL_CONFIGS[model_key]['name']}]"
        ):
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"]

            # Extract filename and original NIfTI affine matrix (compatible with both MetaTensor and meta_dict)
            if "image_meta_dict" in batch:
                filepath = batch["image_meta_dict"]["filename_or_obj"][0]
                original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            else:
                filepath = batch["image"].meta["filename_or_obj"][0]
                original_affine = batch["label"].meta["affine"][0].numpy()

            case_name = Path(filepath).name
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)

            # 1. Sliding window inference in resampled (1.5 x 1.5 x 2.0 mm) space
            logits = sliding_window_inference(
                inputs=val_inputs,
                roi_size=PREPROCESS_CONFIG["roi_size"],
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=infer_overlap,
                mode="gaussian",
            )

            # 2. Softmax + Argmax -> discrete class labels [0..13]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_resampled = np.argmax(probs, axis=1).astype(np.uint8)[0]

            # 3. Resample 3D prediction back to original label shape
            gt_mask = val_labels.cpu().numpy()[0, 0, :, :, :]
            pred_native = resample_3d(pred_resampled, target_shape)

            # 4. Save predicted NIfTI volume in original patient space
            if save_predictions:
                out_path = out_dir / case_name
                nib.save(
                    nib.Nifti1Image(pred_native.astype(np.uint8), original_affine),
                    str(out_path),
                )

            # 5. Compute organ-wise Dice (just like official test.py)
            organ_dices = {}
            for organ_idx, organ_name in BTCV_ORGANS.items():
                organ_dices[organ_name] = compute_dice(
                    pred_native == organ_idx, gt_mask == organ_idx
                )

            mean_case_dice = float(np.mean(list(organ_dices.values())))
            case_results.append(
                {
                    "case": case_name,
                    "mean_dice": mean_case_dice,
                    **organ_dices,
                }
            )
            print(f"  {case_name}: Mean Dice = {mean_case_dice * 100:.2f}%")

    overall_mean_dice = float(np.mean([r["mean_dice"] for r in case_results]))
    print(
        f"==> [{MODEL_CONFIGS[model_key]['name']}] Overall Mean Dice: {overall_mean_dice * 100:.2f}%"
    )
    return {
        "model_key": model_key,
        "infer_overlap": infer_overlap,
        "overall_mean_dice": overall_mean_dice,
        "cases": case_results,
    }
