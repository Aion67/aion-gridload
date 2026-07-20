# aion-gridload

Training and evaluation of a machine learning model to predict grid load.

## Repository Layout

This repository follows the workflow plan in `docs/workflow.md`.

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── experiments/
├── reports/
│   └── figures/
├── src/
├── best_model/
└── docs/
```

## Results

Final model: **GradientBoostingRegressor** — test **RMSE 12.86 MW, R² 0.984, MAPE 1.71%**.
Top load driver: the population × industrial interaction. Full write-up in
[reports/final_report.md](reports/final_report.md); every experiment in
[experiment_log.md](experiment_log.md).

## Setup & Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run notebooks in order (each is already executed with outputs saved):

1. `notebooks/00_data_understanding.ipynb` — missing values & data quality (Tasks 1–3)
2. `notebooks/01_eda.ipynb` — visualizations, writes `data/processed/grid_load_clean.csv` (Tasks 7–9)
3. `notebooks/02_baseline_models.ipynb` — model screening (Task 10)
4. `notebooks/experiments/` — imputation, feature engineering, CV selection, explainability (Tasks 2–6, 11)

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_understanding.ipynb
```

## Layout notes

- Raw data stays immutable in `data/raw/`; cleaned data lands in `data/processed/`.
- Reusable logic lives in `src/` (loading, preprocessing, features, modeling, evaluation).
- Report-ready plots are in `reports/figures/`; the saved model in `best_model/`.

## Key Docs

- `docs/workflow.md` for the team workflow and folder structure.
- `docs/grid-load-summary.md` for the project tasks and deliverables.
- `docs/group_D.md` for the Group D member roster.
