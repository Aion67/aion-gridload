# 04 — Actionable Insights — Learnings

**Notebook:** `notebooks/04_actionable_insights.ipynb` · **Task 12**

## What we did
Interpreted the final GradientBoosting model into business insights.

## Evidence
- **Load drivers** ([04_load_drivers.png](../figures/04_load_drivers.png)): impurity importance `IndustrialIndex` 0.75, `PopulationIndex` 0.18 → the two account for **~93%** of predictive power; permutation importance agrees.
- **When/where peaks** ([04_when_where_peaks.png](../figures/04_when_where_peaks.png)): daily curve is non-flat; **Central** region highest (648 MW vs ~603–612 elsewhere).
- **Weekend vs weekday hourly** ([04_hourly_weekend_vs_weekday.png](../figures/04_hourly_weekend_vs_weekday.png)): overall means near-identical (615 vs 616 MW) but the **shape diverges — up to 33 MW at hour 16** — so daily *pattern* matters more than the weekday/weekend average.
- **Model error** ([04_error_breakdown.png](../figures/04_error_breakdown.png)): worst region **Eastern**, worst hours **21, 6, 17**.
- **Sensitivity** ([04_sensitivity.png](../figures/04_sensitivity.png)): predicted load rises **+3.6 MW per unit** of IndustrialIndex, **+2.6 MW per unit** of PopulationIndex.

## Learnings / recommendations
1. **Protect the two activity indices** — they carry ~93% of the signal; degrade most when stale/missing.
2. **Plan capacity around the evening peak and Central region.**
3. **De-prioritise weather feeds** — minor predictors here.
4. **Validate data at entry** (the raw data had impossible values).
5. **Widen safety margins** for the high-error segments (Eastern; hours 21/6/17).

## Limitations
Cross-sectional data (predicts from conditions, not a time-series forecaster); small dataset (~1.5k rows).
