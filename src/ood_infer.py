"""Bonus task (OOD), GPU step: Swin UNETR (BTCV weights) inference on TotalSegmentator CTs.

Standalone on purpose (no imports from `src`), so it can be copied to a Colab GPU VM on its own.
Each CT is loaded and preprocessed once (official BTCV test preprocessing + RAS orientation) and run through
all requested models. The argmax prediction is saved in the model's working grid (1.5 x 1.5 x 2.0 mm, RAS)
together with that grid's affine; resampling back to the label grid and the metrics are done locally by
`src/ood_eval.py`.

Usage (on the GPU machine):
    pip install monai==1.5.1
    python ood_infer.py --cases abdominal.txt --root /content/ts/Totalsegmentator_dataset_v300 \
        --ckpt-dir /content/ckpt --out /content/pred_raw
Checkpoints: the three `swin_unetr.*_pretrained.pt` files from src/config.py (MODEL_CONFIGS[...]["url"]).
Resumable: cases that already have a prediction for every model are skipped.

Notes for reproducibility:
- fp32 inference (no autocast), like the official test script. On Ampere GPUs (A100) PyTorch runs
  convolutions in TF32 by default; on two cases run on both a T4 (fp32) and an A100 (TF32) < 0.05 % of
  foreground voxels differed (data/totalsegmentator_ood/provenance/tf32_check).
- sw_batch_size only changes how many windows run in parallel, not the result.
- For large scans, the full-size 14-channel output is stitched in CPU RAM instead of GPU memory
  (same numbers; avoids out-of-memory on 15 GB GPUs).
"""
import argparse
import inspect
import os
import time

import nibabel as nib
import torch
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

CKPT = {
    "base": (48, "swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"),
    "small": (24, "swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt"),
    "tiny": (12, "swin_unetr.tiny_5000ep_f12_lr2e-4_pretrained.pt"),
}


def load_model(key: str, ckpt_dir: str, device: torch.device) -> SwinUNETR:
    feature_size, fname = CKPT[key]
    kw = dict(in_channels=1, out_channels=14, feature_size=feature_size, drop_rate=0.0, attn_drop_rate=0.0,
              dropout_path_rate=0.0, use_checkpoint=False)
    if "img_size" in inspect.signature(SwinUNETR.__init__).parameters:  # MONAI < 1.3
        kw["img_size"] = (96, 96, 96)
    model = SwinUNETR(**kw)
    ck = torch.load(os.path.join(ckpt_dir, fname), map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("state_dict", ck))
    return model.eval().to(device)


def get_transforms():
    # Official BTCV test preprocessing (spacing 1.5 x 1.5 x 2.0 mm, HU window [-175, 250] -> [0, 1]).
    # Orientationd(RAS) is added because TotalSegmentator volumes are not stored in BTCV's native orientation
    # (the official training transforms reorient to RAS as well).
    return transforms.Compose([
        transforms.LoadImaged(keys="image"),
        transforms.EnsureChannelFirstd(keys="image"),
        transforms.Orientationd(keys="image", axcodes="RAS"),
        transforms.Spacingd(keys="image", pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        transforms.ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="text file with one case id (e.g. s0013) per line")
    ap.add_argument("--root", default="/content/ts/Totalsegmentator_dataset_v300")
    ap.add_argument("--ckpt-dir", default="/content/ckpt")
    ap.add_argument("--out", default="/content/pred_raw")
    ap.add_argument("--models", nargs="+", default=["base", "small", "tiny"])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--sw-batch", type=int, default=8)
    a = ap.parse_args()

    device = torch.device("cuda")
    cases = [line.strip() for line in open(a.cases) if line.strip()]
    todo = [c for c in cases if not all(os.path.exists(f"{a.out}/{k}/{c}.nii.gz") for k in a.models)]
    for k in a.models:
        os.makedirs(f"{a.out}/{k}", exist_ok=True)
    models = {k: load_model(k, a.ckpt_dir, device) for k in a.models}
    print(f"cases={len(cases)} todo={len(todo)} models={a.models} overlap={a.overlap} fp32", flush=True)

    ds = Dataset([{"image": f"{a.root}/{c}/ct.nii.gz", "case": c} for c in todo], transform=get_transforms())
    # 2 workers preprocess the next CTs on the CPU while the GPU runs the current one
    dl = DataLoader(ds, batch_size=1, num_workers=2, prefetch_factor=2)
    # full-size output fits on the GPU up to ~20M voxels on a 15 GB T4 (~54M on a 40 GB A100)
    max_vox = torch.cuda.get_device_properties(0).total_memory / 750

    for i, b in enumerate(dl):
        t0 = time.time()
        case = b["case"][0]
        img = b["image"].to(device)
        affine = b["image"].affine[0].numpy() if b["image"].affine.ndim == 3 else b["image"].affine.numpy()
        out_dev = device if img[0, 0].numel() < max_vox else torch.device("cpu")
        times = []
        for k, m in models.items():
            t1 = time.time()
            with torch.no_grad():
                logits = sliding_window_inference(img, (96, 96, 96), a.sw_batch, m, overlap=a.overlap,
                                                  mode="gaussian", sw_device=device, device=out_dev)
            pred = torch.argmax(logits, 1)[0].to(torch.uint8).cpu().numpy()
            nib.save(nib.Nifti1Image(pred, affine), f"{a.out}/{k}/{case}.nii.gz")
            times.append(f"{k}={time.time() - t1:.0f}s")
            del logits
        print(f"{i + 1}/{len(todo)} {case} grid={tuple(img.shape[2:])} {' '.join(times)} "
              f"total={time.time() - t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
