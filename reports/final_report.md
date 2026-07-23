# Grid Load Prediction — Final Report

**Group D — BSE2301 Software Engineering Mini Project 2 (2026)**

End-to-end summary from data understanding to actionable insights. Each section links its
evidence (figures in `figures/`) and its detailed per-notebook write-up.

---

## 1. Introduction
Goal: predict Uganda grid load (`GridLoad_MW`) from regional, temporal, weather and economic
indicators. The data is **cross-sectional tabular** (each row = one region/hour observation), so
this is **supervised regression**, not time-series forecasting. Dataset: 1490 rows × 11 columns.

## 2. Data understanding & cleaning (Tasks 1–3)
*Detail: [00_data_understanding/](00_data_understanding/README.md)*

The raw data had two kinds of dirt: placeholder tokens (`"N/A"`, `"###"`, `"hot"`, `"MW900"`) and
physically-impossible values (`Hour=30`, negative humidity). We coerced tokens to NaN,
range-checked numeric columns, dropped 10 duplicates, and median-imputed the rest.

- **Result:** invalid cells **15 → 0**, rows **1490 → 1480**, zero missing.
- **Why median:** missingness <1% per column; median is robust to outliers and reproducible.

## 3. Exploratory data analysis (Tasks 7–9)
*Detail: [01_eda/](01_eda/README.md)*

- Target ~370–900 MW, mean ~616 ([target_distribution.png](figures/target_distribution.png)).
- **Load is driven by activity, not weather** ([correlation_heatmap.png](figures/correlation_heatmap.png)):
  `IndustrialIndex` r = **0.86**, `PopulationIndex` r = **0.44**; all others |r| < 0.13.
- Clear daily curve ([load_by_hour_and_day.png](figures/load_by_hour_and_day.png)); Central region
  highest ([load_by_region.png](figures/load_by_region.png)).

## 4. Feature engineering (Tasks 4–6)
*Detail: [02_feature_eng/](02_feature_eng/README.md) · ablation: [feature_ablation_01/](feature_ablation_01/README.md)*

Dropped `Record_ID`; added peak-hour flags (`Morning_Peak`, `Evening_Peak`), calendar flags
(`Is_Weekend`, `Is_Working_Day`), and `Region_Encoded`. Feature count 11 → 15.

**Ablation finding** ([ablation_base_vs_full.png](figures/ablation_base_vs_full.png)) — the flags'
value is model-dependent:

| Model | improvement from flags |
| --- | --- |
| Linear / Ridge | **~12%** (RMSE 21.0 → 18.5) |
| Trees (GBM) | ~0% (already split on raw Hour/DayOfWeek) |

We keep them: big help for linear models, harmless for trees, better interpretability.

## 5. Model training, CV & overfitting (Tasks 10–11)
*Detail: [03_model_training_eval/](03_model_training_eval/README.md)*

Six candidates, 5-fold CV ([03_cv_model_comparison.png](figures/03_cv_model_comparison.png)):

| Model | CV RMSE (MW) | CV R² | overfit gap |
| --- | --- | --- | --- |
| **GradientBoosting** | **16.0** | 0.975 | +3.5 (healthy) |
| Linear / Ridge | 18.5 | 0.966 | ~0 |
| RandomForest | 23.1 | 0.948 | +12.7 (overfit) |
| DecisionTree | 35.1 | 0.882 | +16.5 (overfit) |
| Baseline (mean) | 102.5 | ~0 | — |

Overfitting was checked explicitly via train-vs-test gap
([03_overfitting_gap.png](figures/03_overfitting_gap.png)) and a learning curve
([03_learning_curve.png](figures/03_learning_curve.png)). RandomForest/DecisionTree fit the
training data almost perfectly but generalise worse → rejected.

## 6. Results
**GradientBoosting** selected and saved to `best_model/`. Held-out test:

| Metric | Value |
| --- | --- |
| RMSE | **13.96 MW** |
| R² | **0.982** |
| MAPE | **1.88 %** |

Predictions sit tight to the diagonal ([03_best_actual_vs_pred.png](figures/03_best_actual_vs_pred.png)).

## 7. Actionable insights & recommendations (Task 12)
*Detail: [04_actionable_insights/](04_actionable_insights/README.md)*

- **Drivers** ([04_load_drivers.png](figures/04_load_drivers.png)): `IndustrialIndex` + `PopulationIndex` ≈ **93%** of predictive power.
- **Sensitivity** ([04_sensitivity.png](figures/04_sensitivity.png)): +3.6 MW per unit IndustrialIndex, +2.6 MW per unit PopulationIndex.
- **When/where** ([04_when_where_peaks.png](figures/04_when_where_peaks.png), [04_hourly_weekend_vs_weekday.png](figures/04_hourly_weekend_vs_weekday.png)): Central region highest; weekday/weekend averages tie (615 vs 616 MW) but the hourly *shape* diverges up to 33 MW at hour 16.
- **Error hotspots** ([04_error_breakdown.png](figures/04_error_breakdown.png)): Eastern region; hours 21, 6, 17.

**Recommendations:**
1. Protect the two activity indices — collect them accurately and on time.
2. Plan capacity around the evening peak and Central region.
3. De-prioritise weather data feeds (minor predictors).
4. Validate data at entry to block impossible values.
5. Apply wider safety margins in the high-error segments.

## 8. Conclusions & limitations
GradientBoosting predicts grid load with **R² 0.982 / MAPE 1.88%** — accurate enough for planning
support — and its drivers are interpretable (industrial + population activity). Limitations:
cross-sectional data (not a true forecaster), small dataset (~1.5k rows). Next steps: a real
timestamp would enable lag/rolling forecasting features; hyperparameter tuning could add marginal
gains.

---

*Per-notebook learnings: [00](00_data_understanding/README.md) · [01](01_eda/README.md) ·
[02](02_feature_eng/README.md) · [ablation](feature_ablation_01/README.md) ·
[03](03_model_training_eval/README.md) · [04](04_actionable_insights/README.md). Full experiment
table: [`experiment_log.md`](../experiment_log.md).*
