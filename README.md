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

## Start Here

1. Put the original dataset in `data/raw/`.
2. Save cleaned or transformed datasets in `data/processed/`.
3. Keep exploratory notebooks in `notebooks/` and experiment notebooks in `notebooks/experiments/`.
4. Move reusable code into `src/`.
5. Store report-ready plots in `reports/figures/`.
6. Keep the final model and its notes in `best_model/`.

## Key Docs

- `docs/workflow.md` for the team workflow and folder structure.
- `docs/grid-load-summary.md` for the project tasks and deliverables.
- `docs/group_D.md` for the cleaned Group D member roster.
