# FoundationsAIMB_Mini-Project_Segmentation

Mini-Project 1 (Segmentation) for **Foundations of AI in Biomedicine**.

## Project Goal

Take a published segmentation model (Swin UNETR), run its pretrained weights on abdominal CT scans,
and check whether we get the same results as the paper. No training needed.

- **Paper:** Tang et al., [*Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis*](https://openaccess.thecvf.com/content/CVPR2022/html/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.html), CVPR 2022 (Swin UNETR)
- **Official code:** [MONAI research-contributions – SwinUNETR/BTCV](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)
- **Dataset:** Beyond the Cranial Vault (BTCV) multi-organ abdominal CT (13 organs)

## Tasks

| Task | Points | Description |
|---|---|---|
| 1. Understand the Method | 15 | Explain the problem Swin UNETR solves, its main architectural components, how a Swin Transformer encoder differs from a CNN encoder, and why self-supervised pre-training helps 3D medical segmentation. |
| 2. Reproduce BTCV Results | 30 | Run inference on BTCV with the pretrained **Swin UNETR Base, Small and Tiny** models. No training required. |
| 3. Quantitative Evaluation | 25 | For each model, compute **Dice**, **Hausdorff distance** and **organ-wise performance**. Report overall and per-organ results and discuss them: best model, easiest/hardest organs, whether Dice and HD agree, plus one very good and one very bad example. |
| 4. Qualitative Evaluation | 10 | Visualise a good case, a poor case and typical segmentation errors, and discuss possible causes of failure. |
| 5. Reproducibility Analysis | 20 | Compare with the published results: how close they are, possible causes of differences (preprocessing, splits, software versions, inference settings, metrics) and how well the method is documented. |
| Bonus: OOD Evaluation | 15 | Evaluate the pretrained model, without fine-tuning, on an out-of-distribution dataset (e.g. TotalSegmentator) using structures compatible with BTCV, and analyse robustness to the dataset shift. |

## Report

- Submit as **PDF**. The first page lists all group members' names and matriculation numbers.
- Structure it by Tasks 1–5, with the bonus task as an extra section.
- Show understanding and critical analysis, not just a list of numbers.
- **Due: Friday, 20.11.2026, 11:59 pm.**

Full assignment: [`docs/AIM_1_Elite_P1.pdf`](docs/AIM_1_Elite_P1.pdf)

## Repository Structure

```
docs/        Assignment description, report notes
notebooks/   02_btcv_inference.ipynb    – Task 2: inference on BTCV
             03_quantitative_eval.ipynb – Task 3: Dice, HD95, HD, organ-wise results
             04_visual_inspection.ipynb – Task 4: good/poor case and typical errors
src/         config, data loading, weight download, inference (Dice), evaluate (Hausdorff),
             visualization (Task 4 plots)
results/     metrics/ (CSV results), figures/ (plots), predictions/ (git-ignored)
```

## Setup

Tested with Python 3.14.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Data (manual):** BTCV is not in this repo because its license does not allow redistribution and the files are large.

1. Create a free account at [synapse.org](https://www.synapse.org) and join the
   [BTCV challenge](https://www.synapse.org/#!Synapse:syn3193805).
2. Download **`RawData.zip`** (Files → Abdomen).
3. Unzip it and rename the folders:

```
data/btcv/imagesTr/   # from Training/img/
data/btcv/labelsTr/   # from Training/label/
```

**Weights (automatic):** the notebook downloads the Base, Small and Tiny checkpoints.

**Run Task 2:**

```bash
jupyter notebook notebooks/02_btcv_inference.ipynb
```

On CPU this takes about 1.5 hours in total. Predicted masks are saved to `results/predictions/`.
Set `INFER_OVERLAP` in the notebook to `0.5` (official default, used in Tasks 3 and 4) or `0.7`.

**Run Task 3:** needs the BTCV data (see above), `data/btcv/dataset_0.json` (downloaded by the Task 2 notebook)
and the Task 2 predictions (run Task 2 first, or get the `results/predictions/` folder from a teammate).

```bash
jupyter notebook notebooks/03_quantitative_eval.ipynb
```

Takes a few minutes. Results go to `results/metrics/task3_*.csv` and `results/figures/`.

**Run Task 4:** needs the same inputs as Task 3 plus the Task 3 metric CSVs.

```bash
jupyter notebook notebooks/04_visual_inspection.ipynb
```

Figures go to `results/figures/task4_*.png`.
