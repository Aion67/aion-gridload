# Best Model

Final chosen model for grid load prediction. Selected in
`notebooks/experiments/model_02_cv_selection.ipynb`, tuned in
`notebooks/experiments/tuning_01_gradient_boosting.ipynb`, and produced by the consolidated
`final_best_model.ipynb` in this folder (raw → clean → features → tuned GBM, with visuals).

| File | Contents |
| --- | --- |
| `best_model.joblib` | Fitted **tuned** `GradientBoostingRegressor` (refit on all cleaned data) |
| `metrics.csv` | Held-out test metrics |
| `feature_names.csv` | Exact feature column order the model expects |

Tuned config: `learning_rate=0.1, max_depth=2, n_estimators=400, subsample=0.8, min_samples_leaf=5`.

## Performance (held-out test)

| Metric | Value |
| --- | --- |
| RMSE | 10.13 MW |
| MAE | 8.32 MW |
| R² | 0.990 |
| MAPE | 1.40 % |

## Load a saved model

```python
import joblib, pandas as pd
from src.data_loading import PROCESSED_DIR
from src.features import build_features, split_xy

model = joblib.load("best_model/best_model.joblib")
df = pd.read_csv(PROCESSED_DIR / "grid_load_clean.csv")
X, y = split_xy(build_features(df))
preds = model.predict(X)
```

Feature engineering must match `src/features.py::build_features` (column order in
`feature_names.csv`).
