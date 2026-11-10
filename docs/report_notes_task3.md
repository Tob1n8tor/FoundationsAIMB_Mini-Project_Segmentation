# Task 3 – Notes for the report (temporary)

## Metrics
- **Dice**: how much the prediction overlaps the true organ (1 = perfect).
- **HD95**: how far the predicted outline is from the true one, in mm (lower = better).
- Organ missing in prediction or ground truth → HD can't be computed (left out of averages).

## Overall results

| Model | Dice [%] | Paper Dice [%] | HD95 median [mm] |
|---|---|---|---|
| Tiny | 66.31 | 70.35 | 26.4 |
| Small | 77.89 | 79.34 | 12.4 |
| Base | 81.37 | 81.86 | 6.4 |

- **Base is the best model.** Bigger model = better results.

## Organs
- **Easiest:** liver (0.96), spleen (0.95), kidneys (0.93) – large and clearly visible.
- **Hardest:** adrenal glands (0.64–0.69), esophagus (0.75), gallbladder (0.77) – small or thin.
- Model size matters most for small organs: Tiny never finds the right adrenal gland (Dice 0).

## Best and worst example

| | Model / scan | Dice | HD95 median |
|---|---|---|---|
| Best | Base / img0039 | 83.8% | 7.4 mm |
| Worst | Tiny / img0038 | 60.2% | 222.6 mm |

## Do Dice and HD agree?
- **Not always.** They rank the models the same way, but per organ they barely relate.
- Reason: models predict small **false spots far from the organ**. Dice hardly changes, HD becomes very large.
- Example: liver has the best Dice (0.96) but a high HD95 because of these spots.
- → Use both metrics: Dice for overlap, HD for outline errors.
