# 03 — Model Training & Evaluation — Learnings

**Notebook:** `notebooks/03_model_training_eval.ipynb` · **Tasks 10–11 + overfitting**

## What we did
Trained 6 candidates on the engineered features, cross-validated, and checked for overfitting.

## Evidence
- **5-fold CV RMSE** ([03_cv_model_comparison.png](../figures/03_cv_model_comparison.png)):

  | Model | CV RMSE (MW) | CV R² |
  | --- | --- | --- |
  | **GradientBoosting** | **16.0** | 0.975 |
  | Linear / Ridge | 18.5 | 0.966 |
  | RandomForest | 23.1 | 0.948 |
  | DecisionTree | 35.1 | 0.882 |
  | Baseline (mean) | 102.5 | ~0 |

- **Overfitting — train vs test gap** ([03_overfitting_gap.png](../figures/03_overfitting_gap.png)): DecisionTree **+16.5**, RandomForest **+12.7** (memorising), **GradientBoosting +3.5 (healthy)**, linear ~0.
- **Learning curve** ([03_learning_curve.png](../figures/03_learning_curve.png)): train/CV converge → enough data.
- **Held-out test (GBM):** RMSE **13.96**, R² **0.982**, MAPE **1.88%** ([03_best_actual_vs_pred.png](../figures/03_best_actual_vs_pred.png)).
- Feature-vs-target correlation ([03_feature_target_corr.png](../figures/03_feature_target_corr.png)).

## Learnings / decisions
- Every model crushes the mean baseline → features carry real signal.
- **RandomForest/DecisionTree rejected for overfitting** despite fitting train well.
- **GradientBoosting selected** (best CV, healthy gap) and saved to `best_model/`.
