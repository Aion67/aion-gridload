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

## Final model

**GradientBoostingRegressor** — held-out test: **RMSE 12.86 MW, MAE 10.23, R² 0.984, MAPE 1.71%**.
Saved: `best_model/best_model.joblib` (+ `metrics.csv`, `feature_names.csv`).

Update this file whenever you finish a notebook or a model run.
