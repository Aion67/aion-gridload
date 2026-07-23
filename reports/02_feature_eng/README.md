# 02 — Feature Engineering — Learnings

**Notebook:** `notebooks/02_feature_eng.ipynb` · **Tasks 4–5**

## What we did
Turned the cleaned columns into model-ready features.

## Evidence / changes
- Dropped `Record_ID` (an index, no predictive value).
- Built calendar flags from `Hour`/`DayOfWeek`:
  - `Morning_Peak` (hours 6–9), `Evening_Peak` (hours 17–20)
  - `Is_Weekend` (Sat/Sun), `Is_Working_Day`
- Encoded `Region` → `Region_Encoded` with `LabelEncoder` (Central=0, Eastern=1, Northern=2, Western=3).
- Feature count: **11 → 15**.

## Learnings / decisions
- The EDA daily curve motivated explicit peak-hour flags so **linear** models can capture the non-linear hour effect (trees can already split on raw `Hour`).
- **Decision:** retain the engineered features; their real value is quantified later in `feature_ablation_01` (they help linear models ~12%, trees ~0%).

## Feeds
Feature recipe reused inline by `03_model_training_eval`, which saves `data/processed/grid_load_features.csv`.
