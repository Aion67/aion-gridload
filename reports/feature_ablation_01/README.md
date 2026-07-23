# Feature Ablation — Learnings

**Notebook:** `notebooks/experiments/feature_ablation_01.ipynb` · **Task 6**

## What we did
Tested whether the engineered peak/weekend flags actually help, by training every candidate **with vs without** them (5-fold CV).

## Evidence ([ablation_base_vs_full.png](../figures/ablation_base_vs_full.png))

| Model | base RMSE | full RMSE | improvement |
| --- | --- | --- | --- |
| Linear / Ridge | 21.0 | 18.5 | **~12%** |
| DecisionTree | 34.1 | 33.2 | 2.7% |
| RandomForest | 23.5 | 23.1 | 1.6% |
| GradientBoosting | 16.00 | 15.99 | 0.06% |

## Learnings / decisions
- The flags **help linear models a lot** (they hand over the non-linear peak-hour signal a line can't express) but are **near-redundant for trees**, which already split on raw `Hour`/`DayOfWeek`.
- **Decision:** keep the flags — harmless for the chosen tree model, valuable for interpretability and for the linear baselines. The dominant signal is still `IndustrialIndex`/`PopulationIndex`.
- Value of engineered features is **model-dependent** — always ablate, don't assume.
