# Task 3 – Notes for the report

> **Task:** For each evaluated model, compute Dice score, Hausdorff distance and organ-wise performance.
> Report both the overall performance and the performance for individual organs.

All numbers below are printed by [`notebooks/03_quantitative_eval.ipynb`](../notebooks/03_quantitative_eval.ipynb),
which follows the structure of the report. Metrics are computed by `src/evaluate.py`.

## Setup
- 3 models (Tiny, Small, Base) × 6 validation scans (img0035–img0040) × 13 organs, overlap 0.5 and 0.7.
- **Dice**: official `dice()`; reported in %.
- **HD95**: 95th percentile of surface distances in each direction, then the larger one (MONAI convention), in mm.
  **HD**: the same with the maximum instead of the 95th percentile.
- **Aggregation**: mean over the 13 organs per scan, then over the 6 scans. HD95 median: over all organ–scan pairs.
- **Absent organ** (gallbladder in img0035): Dice = 0 in the overall mean, as in the official code; excluded
  in the organ-wise table. **Missed organ** (Tiny, right adrenal gland, 6 pairs): Dice 0, HD95 undefined.

## Overall results (Table 2)

| Overlap | Model | Dice [%] | Official [%] | HD95 mean [mm] | HD95 median [mm] | Missed |
|---|---|---|---|---|---|---|
| 0.5 | Base | 81.37 | 81.86 | 44.0 | 6.4 | 0 |
| 0.5 | Small | 77.89 | 79.34 | 112.8 | 12.4 | 0 |
| 0.5 | Tiny | 66.31 | 70.35 | 104.5 | 26.4 | 6 |
| 0.7 | Base | 81.69 | 82.25 | 8.3 | 5.0 | 0 |
| 0.7 | Small | 78.95 | 79.79 | 58.6 | 9.0 | 0 |
| 0.7 | Tiny | 66.62 | 72.05 | 52.7 | 17.4 | 6 |

File: `results/metrics/task3_overall_overlap_comparison.csv`

## Key points
- **Best model:** Base at both overlaps (Base > Small > Tiny in all 6 scans at 0.5, Wilcoxon p = 0.031).
  Only Base is self-supervised pre-trained, so model size and pre-training cannot be separated.
- **Organs:** easiest are liver, spleen, kidneys and aorta (large, high contrast); hardest are the adrenal glands,
  esophagus, portal/splenic veins and gallbladder (small or thin). Table 3:
  `results/metrics/task3_organwise_overlap_comparison.csv`.
- **Examples (Table 4):** very good Base/img0039 (83.82 %), very bad Tiny/img0038 (60.18 %),
  additional Base/img0035 (77.01 %, absent gallbladder).
- **Dice vs. HD95:** same model ranking (Dice and median HD95), but different conclusions per organ.
  Large HD95 values for well-segmented organs come from small distant fragments, mostly in air outside the patient.
  Overlap 0.7 halves the fragments (Base: 617 → 306) and lowers Base's mean HD95 from 44.0 to 8.3 mm,
  while Dice changes by only +0.3 points. Files: `task3_fragments_overlap_comparison.csv`,
  `results/figures/task3_dice_vs_hd95.png`.

`task3_overall_overlap0.5.csv` and `task3_organwise_overlap0.5.csv` are older files kept for reference; the notebook
no longer writes them.
