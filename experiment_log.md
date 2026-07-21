# Experiment Log

Track every meaningful experiment here so the final report can be written from the log instead of from memory.

| ID | Notebook | Category | Hypothesis | What changed | Metric before -> after | Decision | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | 00_data_understanding | imputation | Placeholder tokens/out-of-range values are rare enough (<1% per column) that median imputation is a safe default over dropping rows | Coerced dirty tokens (`"hot"`, `"N/A"`, `"unknown"`, `"###"`, `"high"`, `"MW900"`) and out-of-range values (Hour=30, negative Humidity/Rainfall) to NaN, dropped 10 exact duplicate rows, imputed remaining numeric gaps with the column median | missing/invalid cells: 15 -> 0; rows: 1490 -> 1480 | Kept | Jovia |
| 02 | 01_eda | eda | IndustrialIndex and PopulationIndex are the strongest drivers of GridLoad_MW; Hour has a real but cyclical (non-linear) effect | Computed correlation matrix + grouped means by Hour/DayOfWeek/Region on the cleaned dataset | corr(IndustrialIndex, GridLoad_MW) = 0.86, corr(PopulationIndex, GridLoad_MW) = 0.44, all other numeric predictors \|r\| < 0.13 | Kept — flagged IndustrialIndex/PopulationIndex as priority features, Hour for cyclical encoding, DayOfWeek as a weak candidate | Jovia |

Update this file whenever you finish a notebook or a model run.