# Experiment Log

Track every meaningful experiment here so the final report can be written from the log instead of from memory.

All RMSE values are in MW on a fixed 80/20 split (random_state=42) unless noted as CV.

| ID | Notebook | Category | Hypothesis | What changed | Metric before -> after | Decision | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | 00_data_understanding | data | Numeric columns are polluted with text + impossible values | Mapped junk tokens to NaN; flagged out-of-range (Hour=30, humidity=-15, solar=1.8, rainfall=-12) | 4 non-numeric text cells + 4 out-of-range cells found; 1 row dropped (missing target) | Coerce + range-check + impute | — |
| 02 | imputation_01_median_vs_knn | imputation | Imputation method changes downstream RMSE | Compared mean vs median vs KNN(k=5) via a fixed RandomForest | RMSE mean 18.95 / median 18.94 / knn 18.93 | Kept **median** (≈KNN accuracy, robust, cheap, reproducible) | — |
| 03 | feature_engineering_01_impact | feature_engineering | Calendar + interaction features beat raw columns | Added cyclical hour/DoW, calendar flags, Temp×Humidity & Pop×Industrial interactions | RMSE 22.89 (raw) -> 18.94 (full); R2 0.949 -> 0.965 | Kept full engineered set (interactions gave the jump) | — |
| 04 | 02_baseline_models | model | Screen candidates on one split before CV | Ran mean-baseline, Linear, Ridge, DecisionTree, RandomForest, GradientBoosting | Best RMSE 12.76 (Ridge/Linear/GBM ~tie); all beat mean-baseline 101.6 | Take Linear/Ridge/GBM into CV | — |
| 05 | model_02_cv_selection | model | Cross-validation picks the most reliable model | 5-fold KFold RMSE on all candidates; refit winner on all data | CV RMSE: GBM 12.73 ± 0.51, Linear 12.75, Ridge 12.76 | Selected **GradientBoosting**; saved to `best_model/` | — |
| 06 | explainability_01_importance | explainability | Identify the load drivers | Permutation + impurity importance on the saved model | `Pop_x_Industrial` dominates (~0.90 impurity), then IndustrialIndex, Region_Central, evening-peak | Prioritise population/industrial data quality | — |
| 07 | diagnostics_01_overfitting | diagnostics | Which models overfit? | Train-vs-test gap, learning curve, validation curves, residuals, error-by-region/hour, richer metrics (MedAE/MaxErr/ExplVar) | RF overfits (train 6.9 / test 18.9, gap 12.0); GBM gap 4.5; Ridge gap 0.2. Learning curve converged | Drop RF; tune GBM in safe ranges (shallow depth) | — |
| 08 | tuning_01_gradient_boosting | tuning | Tuning + regularisation improves GBM | GridSearchCV (108 configs): lr=0.1, depth=2, n_est=400, subsample=0.8, min_leaf=5 | test RMSE 12.86 -> **10.13**; R2 0.984 -> **0.990**; overfit gap 4.5 -> 3.0 | **Saved tuned model as new best** | — |
| 09 | augmentation_01_noise_injection | augmentation | Augmentation regularises the model | Gaussian jitter (sigma sweep), bootstrap resample, log-target | jitter best 12.74 (≈baseline 12.86), bootstrap 13.50 (worse), log-target 12.65 | No meaningful gain — data already large/clean enough (matches converged learning curve) | — |
| 10 | scaling_01_effect | scaling | Feature scaling changes accuracy | 4 scalers × 7 models, 5-fold CV RMSE | SVR 32→20 (needs scaling); KNN 33–57 (poor); linear ~12.75 flat; trees/GBM ~12.7 flat | **No scaling** for the tree final model; scale only if using SVR/KNN/linear | — |
| 11 | final_best_model | model | Combine every winning choice | Raw→clean(range+median)→engineered features→tuned GBM, no scaling/augmentation | test **RMSE 10.13, R² 0.990, MAPE 1.40%**; CV 10.49 ± 0.30 | Shipped as `best_model/best_model.joblib` | — |

## Final model

**GradientBoostingRegressor (tuned)** — `learning_rate=0.1, max_depth=2, n_estimators=400, subsample=0.8, min_samples_leaf=5`.
Held-out test: **RMSE 10.13 MW, MAE 8.32, R² 0.990, MAPE 1.40%** (improved from the default GBM's 12.86 / 0.984 / 1.71).
Saved: `best_model/best_model.joblib` (+ `metrics.csv`, `feature_names.csv`).

Update this file whenever you finish a notebook or a model run.
