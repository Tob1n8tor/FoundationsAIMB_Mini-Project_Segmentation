# FoundationsAIMB_Mini-Project_Segmentation

Mini-Project 1 (**Segmentation**) for *Foundations of AI in Biomedicine I* — Group B (Tobias Konieczny, Paula Maria Gal, Erik Stoklasa, Isabel Martínez Torres).

## Overview

This repository contains our reproduction and analysis of **Swin UNETR** on the **Beyond the Cranial Vault (BTCV)** multi-organ abdominal CT segmentation benchmark (13 organs), using the official pretrained checkpoints without training.

- **Paper:** Tang et al., [*Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis*](https://openaccess.thecvf.com/content/CVPR2022/html/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.html), CVPR 2022
- **Official code:** [MONAI research-contributions – SwinUNETR/BTCV](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)
- **Dataset:** Beyond the Cranial Vault (BTCV) abdominal CT (24 training / 6 validation scans `img0035`–`img0040` from `swin_unetr_btcv_dataset_0.json`)
- **Assignment sheet:** [`docs/AIM_1_Elite_P1.pdf`](docs/AIM_1_Elite_P1.pdf)

> **Bonus Task (Out-of-Distribution Evaluation):**  
> The bonus task (evaluating the pretrained BTCV models without fine-tuning on the 109 test scans of **TotalSegmentator** v3) lives on the **`task-bonus`** branch. To inspect or run it:
> ```bash
> git checkout task-bonus
> ```

---

## Tasks Covered (`main` branch)

| Task | Description | Where in Repo |
|---|---|---|
| **Task 1** | Understand the Swin UNETR method, architecture, and self-supervised pre-training | Report |
| **Task 2** | Reproduce BTCV segmentation inference with **Base**, **Small**, and **Tiny** checkpoints (`overlap = 0.5` & `0.7`) | [`notebooks/02_btcv_inference.ipynb`](notebooks/02_btcv_inference.ipynb), [`src/inference.py`](src/inference.py) |
| **Task 3** | Quantitative evaluation: Dice score, HD95, max HD, organ-wise breakdown, and good/bad case selection | [`notebooks/03_quantitative_eval.ipynb`](notebooks/03_quantitative_eval.ipynb), [`src/evaluate.py`](src/evaluate.py) |
| **Task 4** | Qualitative visual inspection: CT overlays, error maps, surgical clips, and failure mode analysis | [`notebooks/04_visual_inspection.ipynb`](notebooks/04_visual_inspection.ipynb), [`src/visualization.py`](src/visualization.py) |
| **Task 5** | Reproducibility analysis: comparison with official numbers, library adaptations, and documentation gaps | Report & code adaptations in [`src/`](src/) |
| **Bonus** | Out-of-distribution evaluation on TotalSegmentator (109 scans) | Branch **`task-bonus`** |

---

## Key Reproduced Results (BTCV Validation, 6 Scans)

| Model | Feature Size | Params | SSL Pre-trained | Official Dice (`0.5`) | Ours (`0.5`) | Official Dice (`0.7`) | Ours (`0.7`) | Median HD95 (`0.5`) |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| **Swin UNETR / Base** | 48 | 62.1M | Yes | 81.86% | **81.37%** | 82.25% | **81.69%** | **6.4 mm** |
| **Swin UNETR / Small** | 24 | 15.7M | No | 79.34% | 77.89% | 79.79% | 78.95% | 12.4 mm |
| **Swin UNETR / Tiny** | 12 | 4.0M | No | 70.35% | 66.31% | 72.05% | 66.62% | 26.4 mm |

---

## Repository Structure

```text
docs/
  AIM_1_Elite_P1.pdf                 Assignment description
  report_notes_task3.md              Working notes for Task 3 quantitative analysis
notebooks/
  02_btcv_inference.ipynb            Task 2: BTCV preprocessing, sliding-window inference, Dice
  03_quantitative_eval.ipynb         Task 3: Dice, HD95, HD, organ-wise tables & plots
  04_visual_inspection.ipynb         Task 4: Qualitative overlays, error maps, failure cases
src/
  config.py                          Paths, model configs, BTCV 13 organ labels, hyperparameters
  data.py                            Official MONAI test transforms & dataset loader
  download.py                        Automatic download of dataset_0.json and model checkpoints
  inference.py                       Sliding-window inference, native-grid resampling, Dice
  evaluate.py                        Dice, HD95, and max HD computation in native voxel space
  visualization.py                   CT + mask overlays, error maps, and multi-panel figures
results/
  metrics/                           Exported CSVs (per-case, per-organ, and summary tables)
  figures/                           Generated Task 3 and Task 4 plots
  predictions/                       Predicted NIfTI masks (git-ignored)
data/                                BTCV CT volumes and labels (git-ignored)
checkpoints/                         Downloaded pretrained .pt weights (git-ignored)
requirements.txt                     Python dependencies
```

---

## Setup & Reproduction

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data (Manual Download)

BTCV cannot be redistributed in this repository due to its license and file size:

1. Create a free account at [synapse.org](https://www.synapse.org) and join the [BTCV challenge](https://www.synapse.org/#!Synapse:syn3193805).
2. Download **`RawData.zip`** (*Files → Abdomen*).
3. Unzip it and restructure the folders to match `dataset_0.json`:
   ```text
   data/btcv/imagesTr/   # from Training/img/
   data/btcv/labelsTr/   # from Training/label/
   ```

The split file (`data/btcv/dataset_0.json`) and pretrained weights (`checkpoints/`) are downloaded automatically by the Task 2 notebook.

### 3. Run the Notebooks

1. **Task 2 – Inference:**
   ```bash
   jupyter notebook notebooks/02_btcv_inference.ipynb
   ```
   Set `INFER_OVERLAP` to `0.5` (official default, used in Tasks 3 & 4) or `0.7`. Saves predicted NIfTI masks to `results/predictions/` and per-case Dice CSVs to `results/metrics/`.

2. **Task 3 – Quantitative Evaluation:**
   ```bash
   jupyter notebook notebooks/03_quantitative_eval.ipynb
   ```
   Requires BTCV data, `dataset_0.json`, and Task 2 predictions. Outputs `results/metrics/task3_*.csv` and `results/figures/task3_*.png`.

3. **Task 4 – Qualitative Inspection:**
   ```bash
   jupyter notebook notebooks/04_visual_inspection.ipynb
   ```
   Requires the same inputs plus Task 3 metric CSVs. Outputs `results/figures/task4_*`.
