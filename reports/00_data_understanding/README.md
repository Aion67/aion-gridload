# 00 — Data Understanding — Learnings

**Notebook:** `notebooks/00_data_understanding.ipynb` · **Tasks 1–3**

## What we did
Audited the raw `Uganda_Grid_Load_Dataset.csv` (1490 rows, 11 columns) for missing and invalid data, then cleaned it.

## Evidence
- **Two kinds of dirt:** placeholder tokens in numeric columns (`"N/A"`, `"unknown"`, `"###"`, `"hot"`, `"MW900"`) **and** parseable-but-impossible values (`Hour=30`, negative `Humidity_pct`/`Rainfall_mm`).
- **10 exact duplicate rows** removed.
- Missing/invalid cells: **15 → 0** after cleaning; rows **1490 → 1480**.

## Learnings / decisions
- Missingness is <1% per column, so **median imputation** is safe and preferred over dropping rows (robust to the outliers/skew, cheap to justify, reproducible).
- Range-checking is as important as token-cleaning — an out-of-range `Hour=30` parses fine but is still garbage.
- **Decision:** coerce → range-check → median-impute; cleaned data saved to `data/processed/grid_load_clean.csv`.

## Feeds
Clean dataset consumed by `01_eda`, `02_feature_eng`, `03_model_training_eval`.
