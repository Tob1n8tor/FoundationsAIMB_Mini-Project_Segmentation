from pathlib import Path

# 1. Project Paths (resolved relative to this file so notebooks work from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "btcv"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# 2. Official Dataset Split JSON URL (from MONAI SwinUNETR/BTCV README)
DATASET_JSON_URL = "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/swin_unetr_btcv_dataset_0.json"
DATASET_JSON_PATH = DATA_DIR / "dataset_0.json"

# 3. Official Pretrained Models (Task 2 configurations)
MODEL_CONFIGS = {
    "base": {
        "name": "Swin UNETR / Base",
        "feature_size": 48,
        "params_m": 62.1,
        "ssl_pretrained": True,
        "paper_dice_05": 81.86,
        "paper_dice_07": 82.25,
        "filename": "swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
        "url": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
    },
    "small": {
        "name": "Swin UNETR / Small",
        "feature_size": 24,
        "params_m": 15.7,
        "ssl_pretrained": False,
        "paper_dice_05": 79.34,
        "paper_dice_07": 79.79,
        "filename": "swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt",
        "url": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt",
    },
    "tiny": {
        "name": "Swin UNETR / Tiny",
        "feature_size": 12,
        "params_m": 4.0,
        "ssl_pretrained": False,
        "paper_dice_05": 70.35,
        "paper_dice_07": 72.05,
        "filename": "swin_unetr.tiny_5000ep_f12_lr2e-4_pretrained.pt",
        "url": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.tiny_5000ep_f12_lr2e-4_pretrained.pt",
    },
}

# 4. BTCV 13 Organ Labels (Indices 1..13; 0 is Background)
BTCV_ORGANS = {
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "Inferior Vena Cava",
    10: "Portal and Splenic Veins",
    11: "Pancreas",
    12: "Right Adrenal Gland",
    13: "Left Adrenal Gland",
}

# 5. Official Preprocessing & Inference Hyperparameters (from MONAI test.py)
PREPROCESS_CONFIG = {
    "space_x": 1.5,  # Target voxel spacing along X (mm)
    "space_y": 1.5,  # Target voxel spacing along Y (mm)
    "space_z": 2.0,  # Target voxel spacing along Z (mm)
    "a_min": -175.0,  # Minimum CT Hounsfield Unit (HU) clip value
    "a_max": 250.0,  # Maximum CT Hounsfield Unit (HU) clip value
    "b_min": 0.0,  # Normalized min intensity
    "b_max": 1.0,  # Normalized max intensity
    "roi_size": (96, 96, 96),  # 3D patch window size expected by Swin UNETR
    "sw_batch_size": 4,  # Number of 3D windows processed in parallel
    "infer_overlap": 0.5,  # Default sliding window overlap in test.py (0.5 or 0.7)
}

# 6. Bonus task: out-of-distribution evaluation on the TotalSegmentator dataset (v300, test split)
OOD_DIR = PROJECT_ROOT / "data" / "totalsegmentator_ood"
OOD_DATASET_DIR = OOD_DIR / "Totalsegmentator_dataset_v300"  # <case>/ct.nii.gz, <case>/segmentations/*.nii.gz, meta.csv
OOD_LABEL_DIR = OOD_DIR / "ts_btcv"  # <case>_label.nii.gz: 13 TS masks merged into BTCV label indices
OOD_PRED_RAW_DIR = OOD_DIR / "pred_raw"  # <model>/<case>.nii.gz from the GPU step, in the 1.5x1.5x2 mm model grid
OOD_CASE_LIST_DIR = OOD_DIR / "case_lists"  # abdominal.txt / non_abdominal.txt
OOD_SURVEY_PATH = OOD_DIR / "survey.csv"  # per test case: shape, spacing, orientation, organ voxel counts, metadata

# BTCV label index -> TotalSegmentator mask name (all 13 BTCV organs have a TS counterpart)
TS_TO_BTCV = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "portal_vein_and_splenic_vein",
    11: "pancreas",
    12: "adrenal_gland_right",
    13: "adrenal_gland_left",
}
# A test scan counts as "abdominal" if all of these organs are (at least partly) in the field of view
OOD_CORE_ORGANS = ["liver", "spleen", "kidney_right", "kidney_left", "pancreas", "stomach"]
