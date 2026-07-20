# Best Model

Final chosen model for grid load prediction, produced by
`notebooks/experiments/model_02_cv_selection.ipynb`.

| File | Contents |
| --- | --- |
| `best_model.joblib` | Fitted `GradientBoostingRegressor` (refit on all cleaned data) |
| `metrics.csv` | Held-out test metrics |
| `feature_names.csv` | Exact feature column order the model expects |

## Performance (held-out test)

| Metric | Value |
| --- | --- |
| RMSE | 12.86 MW |
| MAE | 10.23 MW |
| R² | 0.984 |
| MAPE | 1.71 % |

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
