# 01 — Exploratory Data Analysis — Learnings

**Notebook:** `notebooks/01_eda.ipynb` · **Tasks 7–9**

## What we did
Explored the cleaned data: distributions, correlations, and grouped load by hour/region/day.

## Evidence
- **Target** ([target_distribution.png](../figures/target_distribution.png)): `GridLoad_MW` ranges ~370–900 MW (mean ~616), roughly unimodal.
- **Correlation with load** ([correlation_heatmap.png](../figures/correlation_heatmap.png)):
  - `IndustrialIndex` **r = 0.86** (dominant)
  - `PopulationIndex` **r = 0.44**
  - everything else weak: `Humidity_pct` −0.13, `Temperature_C` 0.10, `Hour` 0.07, rest < 0.04.
- **By hour** ([load_by_hour_and_day.png](../figures/load_by_hour_and_day.png)): clear non-flat daily curve.
- **By region** ([load_by_region.png](../figures/load_by_region.png)): Central highest.
- Supporting: [predictor_distributions.png](../figures/predictor_distributions.png), [temperature_vs_load.png](../figures/temperature_vs_load.png), [categorical_counts.png](../figures/categorical_counts.png), interactive [industrial_index_vs_load.html](../figures/industrial_index_vs_load.html), [temp_hour_load_3d.html](../figures/temp_hour_load_3d.html).

## Learnings / decisions
- Load is driven by **activity indices**, not weather — this sets modelling priorities.
- `Hour` has a real but **non-linear/cyclical** effect → worth encoding for linear models.
- `Region` shifts the load level → keep it as a predictor.
