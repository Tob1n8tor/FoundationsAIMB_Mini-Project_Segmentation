from pathlib import Path

from monai import data, transforms
from monai.data import load_decathlon_datalist

from src.config import DATA_DIR, DATASET_JSON_PATH, PREPROCESS_CONFIG


def get_test_transforms(use_orientation: bool = False):
    """
    Build the exact inference preprocessing pipeline from MONAI's SwinUNETR/BTCV/utils/data_utils.py.

    Note for Task 5 (Reproducibility Analysis):
    In the official `data_utils.py`, `Orientationd(keys=["image"], axcodes="RAS")` is commented out
    in `test_transform`, and only `image` is resampled with `Spacingd`, while `label` is kept in its
    native voxel grid so predictions can be resampled back to native patient space.
    """
    # Handle MONAI version difference: AddChanneld (MONAI < 1.2) vs EnsureChannelFirstd (MONAI >= 1.2)
    channel_transform = (
        transforms.EnsureChannelFirstd(keys=["image", "label"])
        if hasattr(transforms, "EnsureChannelFirstd")
        else transforms.AddChanneld(keys=["image", "label"])
    )

    xforms = [
        transforms.LoadImaged(keys=["image", "label"], image_only=False),
        # Reshape (H, W, D) -> (1, H, W, D)
        channel_transform,
    ]

    if use_orientation:
        # Optional for Task 5 analysis: re-orient axes to Right-Anterior-Superior (RAS)
        xforms.append(transforms.Orientationd(keys=["image"], axcodes="RAS"))

    xforms.extend(
        [
            transforms.Spacingd(
                keys="image",
                pixdim=(
                    PREPROCESS_CONFIG["space_x"],
                    PREPROCESS_CONFIG["space_y"],
                    PREPROCESS_CONFIG["space_z"],
                ),
                mode="bilinear",
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=PREPROCESS_CONFIG["a_min"],
                a_max=PREPROCESS_CONFIG["a_max"],
                b_min=PREPROCESS_CONFIG["b_min"],
                b_max=PREPROCESS_CONFIG["b_max"],
                clip=True,
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    return transforms.Compose(xforms)


def get_validation_loader(
    data_dir: Path = DATA_DIR,
    json_path: Path = DATASET_JSON_PATH,
    use_orientation: bool = False,
):
    """
    Load the 6 validation CT volumes defined in `dataset_0.json` ("validation" key).
    """
    val_files = load_decathlon_datalist(
        str(json_path),
        is_segmentation=True,
        data_list_key="validation",
        base_dir=str(data_dir),
    )
    val_ds = data.Dataset(
        data=val_files, transform=get_test_transforms(use_orientation=use_orientation)
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Safe across macOS, Linux, and Jupyter Notebooks
    )
    return val_loader, val_files
