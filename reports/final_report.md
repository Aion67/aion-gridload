# Grid Load Prediction — Final Report

**Group D — BSE2301 Software Engineering Mini Project 2 (2026)**
Members: Alinda Joel, Jovia Minallah Matata, Birungi Hairat Muhamed, Aita Joshua Prince, Naluyange Emilly Shirley

---

## 1. Introduction

Electricity grid operators must anticipate demand to dispatch generation efficiently and
avoid both shortfalls and waste. This project builds a machine-learning model that predicts
**grid load in megawatts (`GridLoad_MW`)** for Uganda from regional, temporal, weather and
economic indicators.

**Dataset:** `Uganda_Grid_Load_Dataset.csv` — 1,490 records, 11 columns. It is
**cross-sectional tabular** data (each row = one region/hour observation with its conditions),
so the task is **supervised regression**, not classical time-series forecasting.

| Column | Meaning |
| --- | --- |
| Region | Central / Eastern / Western / Northern |
| Hour, DayOfWeek | Time-of-day (0–23), day index (0–6) |
| Temperature_C, Humidity_pct, Rainfall_mm | Weather conditions |
| PopulationIndex, IndustrialIndex | Demand-driver indices |
| SolarGenerationIndex | Solar generation availability (0–1) |
| **GridLoad_MW** | **Target** — grid load to predict |

**Objective:** clean the data, engineer informative features, compare models with rigorous
cross-validation, and deliver an interpretable best model plus actionable insights.

---

## 2. Dataset description & quality (Tasks 1–3)

### Before cleaning
The raw file mixes non-numeric junk into numeric columns (`"hot"`, `"###"`, `"unknown"`,
`"N/A"`) and contains physically-impossible values.

- **Missing / junk cells:** small but present in every weather/economic column (worst:
  `Rainfall_mm` 0.20%, `Temperature_C` / `SolarGenerationIndex` 0.13%). One row has a missing
  target and is dropped.
- **Out-of-range values (inconsistent entries):** `Hour = 30`, `Humidity_pct = -15`,
  `Rainfall_mm = -12`, `SolarGenerationIndex = 1.8` — one each. Detected in
  `00_data_understanding.ipynb`.

### Cleaning strategy (Task 2 — justification)
1. **Coerce** every numeric column with `to_numeric(errors="coerce")` → all junk text becomes NaN.
2. **Range-check** against physical bounds → impossible values become NaN.
3. **Drop** rows with a missing target (cannot learn/score them) — 1 row.
4. **Impute** the rest: median for numeric (robust to outliers), mode for categorical.

**Why median over KNN (Task 3, `imputation_01_median_vs_knn.ipynb`):** downstream RMSE was
effectively tied (median 18.94 vs KNN 18.93 vs mean 18.95). Median wins on robustness,
reproducibility and cost with no accuracy penalty.

### After cleaning
1,489 rows × 10 columns, **zero missing**, all values within physical range. Saved to
`data/processed/grid_load_clean.csv`.

---

## 3. Exploratory data analysis (Tasks 7–9)

See `01_eda.ipynb`; figures in `reports/figures/`.

- **Target** (`target_distribution.png`): load ranges ~370–880 MW, roughly unimodal.
- **Correlation** (`correlation_heatmap.png`): `PopulationIndex` and `IndustrialIndex`
  correlate most strongly with load; weather effects are secondary.
- **Daily curve** (`load_by_hour_region.png`): load varies clearly by hour → motivates
  cyclical hour features; **Central** region carries the highest load.
- **Weekend & temperature** (`weekend_and_temperature.png`): visible weekday/weekend shift and
  a temperature relationship, motivating calendar flags and a Temp×Humidity interaction.
- **Interactive** (`plotly_scatter_temp_load.html`, `plotly_3d_load.html`): 2D + 3D views for
  the presentation.

---

## 4. Feature engineering (Tasks 4–6)

Implemented in `src/features.py`, evaluated in `feature_engineering_01_impact.ipynb`:

- **Cyclical** `Hour`/`DayOfWeek` as sin/cos (so hour 23 ≈ hour 0).
- **Calendar flags:** `is_weekend`, `is_daytime`, `is_evening_peak`.
- **Interactions:** `Temp_x_Humidity` (cooling demand), `Pop_x_Industrial` (combined activity).
- **Region one-hot** encoding.

**Impact (Task 6):** RMSE fell **22.89 → 18.94 MW** (R² 0.949 → 0.965) from raw columns to the
full engineered set. The **interaction terms delivered nearly all the gain** — foreshadowing
the explainability finding below.

---

## 5. Model training & validation (Tasks 10–11)

**Basic screen** (`02_baseline_models.ipynb`, 80/20 split): every model crushes the
mean-baseline (RMSE 101.6). Ridge, LinearRegression and GradientBoosting tie near RMSE 12.8;
tree singletons trail.

**5-fold cross-validation** (`model_02_cv_selection.ipynb`):

| Model | CV RMSE (MW) |
| --- | --- |
| **GradientBoosting** | **12.73 ± 0.51** |
| LinearRegression | 12.75 ± 0.43 |
| Ridge | 12.76 ± 0.44 |
| RandomForest | 18.65 ± 0.52 |
| DecisionTree | 27.43 ± 1.62 |
| Baseline (mean) | 102.35 ± 3.75 |

**Selected: GradientBoostingRegressor** — best CV mean, low variance, and captures the
interaction structure. This model is then diagnosed and tuned (below) before being saved.

### 5a. Overfitting diagnostics (`diagnostics_01_overfitting.ipynb`)
Rather than trusting a single test score, we inspected training behaviour:

- **Train vs test gap:** RandomForest is near-perfect on train (RMSE 6.9) but 18.9 on test —
  a gap of 12 MW = **clear overfitting**. GradientBoosting's gap is 4.5; Ridge's is ~0.2.
  RF was therefore dropped. (`diag_train_test_gap.png`)
- **Learning curve** (`diag_learning_curve.png`): train and CV curves converge → the dataset is
  already large enough; collecting more rows would help little. This is why *augmentation*, not
  more data, was the next lever tried.
- **Validation curves** (`diag_validation_curves.png`): CV error stops improving at shallow depth
  and moderate `n_estimators` — these ranges fed the tuning grid.
- **Residuals** (`diag_residuals.png`): centred on zero and roughly patternless; error is fairly
  uniform across regions and hours (`diag_error_breakdown.png`).

### 5b. Hyperparameter tuning (`tuning_01_gradient_boosting.ipynb`)
A 108-config `GridSearchCV` over learning rate, depth, estimators, subsample and
`min_samples_leaf` found `lr=0.1, max_depth=2, n_estimators=400, subsample=0.8,
min_samples_leaf=5`. This **improved CV RMSE 12.67 → 10.40** and simultaneously **reduced the
overfitting gap 4.5 → 3.0** (shallow trees + stochastic subsampling regularise). It beat the
saved model on the held-out test and became the new `best_model/`.

### 5c. Data augmentation (`augmentation_01_noise_injection.ipynb`)
Tested Gaussian jitter (noise sweep), bootstrap resampling and a log-target transform. Best
jitter (σ=0.02) reached 12.74 vs 12.86 baseline — negligible; bootstrap was worse; log-target
gave a tiny gain. **Conclusion: augmentation adds little here**, consistent with the converged
learning curve. Tuning, not augmentation, drove the improvement.

### 5d. Feature scaling / standardization (`scaling_01_effect.ipynb`)
Tested 4 scalers (none, Standard, MinMax, Robust) × 7 models by CV RMSE (`scaling_heatmap.png`,
`scaling_sensitivity.png`). **SVR** is highly scale-sensitive (RMSE ~32 → ~20 when scaled); **KNN**
is sensitive but weak overall; **linear** models barely move; **tree ensembles (RF, GBM) are flat
across all scalers**. Since the final model is GradientBoosting, **no scaling is applied** — it
would add complexity for zero benefit. (Scaling would only matter if an SVR/KNN/linear model were
chosen.)

### 5e. Consolidated final model (`best_model/final_best_model.ipynb`)
One notebook combines every winning choice — range+median cleaning, the full engineered feature
set, the tuned GradientBoosting config, no scaling, no augmentation — as a single raw→prediction
pipeline, evaluated with visuals (`final_actual_vs_pred.png`, `final_feature_importance.png`,
`final_pred_vs_actual_line.png`) and saved as the shipped model.

---

## 6. Evaluation results

Held-out test performance of the final **tuned** model:

| Metric | Default GBM | **Tuned GBM (final)** |
| --- | --- | --- |
| RMSE | 12.86 MW | **10.13 MW** |
| MAE | 10.23 MW | **8.32 MW** |
| R² | 0.984 | **0.990** |
| MAPE | 1.71 % | **1.40 %** |

The tuned model explains ~99% of load variance with average error ~1.4%, while carrying a
smaller train-test gap than the default. `actual_vs_predicted.png` shows points tight to the
diagonal.

**Explainability** (`explainability_01_importance.ipynb`, `feature_importance.png`):
`Pop_x_Industrial` dominates (~0.90 impurity importance), followed by `IndustrialIndex`,
`Region_Central`, and `is_evening_peak`. The combined population×industrial activity is by far
the strongest load driver.

---

## 7. Conclusions & recommendations (Task 12)

**Conclusions**
- The data was corrupted (text + impossible values) but recoverable with type coercion, range
  validation and median imputation; only one row was unusable.
- Feature engineering — especially the **population × industrial interaction** — was the single
  biggest accuracy lever.
- After overfitting diagnostics and grid-search tuning, GradientBoosting predicts grid load with
  **R² 0.990 / MAPE 1.40%**, accurate enough for planning support. RandomForest was rejected for
  overfitting; augmentation gave no meaningful gain on this already-sufficient dataset.

**Actionable insights & recommendations**
1. **Protect the top drivers:** accuracy hinges on population and industrial indices — invest in
   collecting these accurately and on time; the model degrades most when they are stale.
2. **Plan around evening peaks and the Central region**, which the model flags as high-load.
3. **Add validation at data entry** to reject impossible values (negative humidity, Hour>23) at
   the source rather than cleaning them later.
4. **Next steps:** if a true timestamp becomes available, add lag/rolling features for genuine
   forecasting; otherwise explore additional interaction terms and gradient-boosting libraries
   (XGBoost/LightGBM) for marginal gains.

---

## Appendix — Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# run notebooks top-to-bottom: 00 -> 01 -> 02 -> experiments/*
jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_understanding.ipynb
```

Shared logic lives in `src/`; every experiment is logged in `experiment_log.md`.
