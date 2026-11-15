# Task 3 – Notes for the report (initial draft)

> **Task:** For each evaluated model, compute Dice score, Hausdorff distance and organ-wise performance.
> Report both the overall performance and the performance for individual organs.

## Setup
- 3 models (Tiny, Small, Base) × 6 validation scans (img0035–img0040) × 13 organs.
- **Dice**: how much the prediction overlaps the true organ (1 = perfect, 0 = no overlap).
- **HD**: the largest distance between the predicted and true outline, in mm (lower = better).
- **HD95**: like HD, but ignores the worst 5% of points, so one stray voxel matters less. Standard in papers.
- Organ missing from the scan (only the gallbladder in img0035): Dice counts as 0, like the official
  code, so we stay comparable with the paper. HD can't be computed, so it is left out of the HD averages.

## Overall results

| Model | Dice [%] | Paper Dice [%] | HD95 median [mm] | HD95 mean [mm] | HD mean [mm] | Missed organs (of 78) |
|---|---|---|---|---|---|---|
| Tiny | 66.31 | 70.35 | 26.4 | 104.7 | 252.8 | 6 |
| Small | 77.89 | 79.34 | 12.4 | 111.9 | 273.4 | 0 |
| Base | 81.37 | 81.86 | 6.4 | 43.6 | 198.1 | 0 |

- We give the HD95 median as well as the mean, because a few huge values push the mean up.
- HD is much bigger than HD95 because it reports the single worst point, often a small false spot far from the organ.

## Organ results (Base model)

| Organ | Dice | HD95 median [mm] |
|---|---|---|
| Liver | 0.96 | 124.2 |
| Spleen | 0.95 | 110.6 |
| Right Kidney | 0.93 | 2.9 |
| Left Kidney | 0.93 | 4.4 |
| Aorta | 0.90 | 5.0 |
| Inferior Vena Cava | 0.83 | 15.7 |
| Pancreas | 0.82 | 6.8 |
| Stomach | 0.78 | 25.6 |
| Gallbladder | 0.77 | 5.0 |
| Portal and Splenic Veins | 0.76 | 11.3 |
| Esophagus | 0.75 | 6.4 |
| Right Adrenal Gland | 0.69 | 4.5 |
| Left Adrenal Gland | 0.64 | 6.8 |

All 3 models: `results/metrics/task3_organwise_overlap0.5.csv` and `results/figures/task3_organwise.png`.

---

## Discussion

### Which model configuration performs best overall?
**Base.** It has the highest Dice (81.37%, only 0.5 points below the paper) and the lowest HD95 (6.4 mm median).
The bigger the model, the better: Tiny < Small < Base. Tiny also misses organs completely
(the right adrenal gland in all 6 scans), while Small and Base find every organ.

### Are there substantial differences between organs?
**Yes.** For Base, Dice ranges from 0.64 (left adrenal gland) to 0.96 (liver). For Tiny the gap is even
bigger: 0.00 to 0.92. Large organs score high with every model. Small organs depend much more on model size.

### Which organs are easiest and most difficult to segment?
- **Easiest:** liver (0.96), spleen (0.95), kidneys (0.93). They are large, have a clear shape and good
  contrast to the tissue around them.
- **Hardest:** adrenal glands (0.64–0.69), esophagus (0.75), gallbladder (0.77). They are small or thin,
  so they cover only a few voxels, and a one-voxel error at the edge already lowers Dice a lot. They also
  have low contrast to nearby tissue, and their shape varies a lot between patients.

### Do Dice score and Hausdorff distance lead to the same conclusions?
**Across models, yes.** Both rank Base > Small > Tiny.
**Per organ, not always.** The liver has the best Dice (0.96) but one of the worst HD95 values (124 mm).
The kidneys have similar Dice but an HD95 of only about 3–4 mm.
- Reason: the models sometimes predict small **false spots far away from the organ**. Such a spot is tiny,
  so Dice barely changes, but HD measures the furthest point, so it becomes very large.
- Dice tells us how much of the organ is covered, and HD tells us whether there are outline errors or stray spots.
  We need both (see `results/figures/task3_dice_vs_hd95.png`).

### Two examples: one very good, one very bad (average Dice and Hausdorff distance)

| | Model / scan | Dice (mean) | HD95 mean [mm] | HD95 median [mm] | HD mean [mm] |
|---|---|---|---|---|---|
| Very good | Base / img0039 | 83.8% | 86.5 | 7.4 | 351.4 |
| Very bad | Tiny / img0038 | 60.2% | 199.7 | 222.6 | 364.2 |

- **Good case:** most organs are segmented well (HD95 median 7.4 mm). The HD95 mean is still high because
  of a few organs with far-away false spots, so the mean alone looks worse than the case really is.
- **Bad case:** low Dice, and even the median HD95 is over 200 mm, so most organs have large outline errors.
  Tiny also misses the right adrenal gland completely here.