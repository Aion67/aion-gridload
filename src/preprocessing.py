"""Data cleaning and preprocessing helpers.

The raw grid-load CSV contains corrupted numeric cells (text like ``"hot"``,
placeholder ``"###"``, ``"unknown"``). These helpers coerce columns to their
intended types, report missingness, and impute gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loading import CATEGORICAL_COLS, NUMERIC_COLS, TARGET

# Physically-valid ranges. Values outside these are data-entry errors (e.g.
# negative humidity, Hour=30, SolarGenerationIndex>1) and are treated as
# missing so they get imputed rather than poisoning the model.
VALID_RANGES = {
    "Hour": (0, 23),
    "DayOfWeek": (0, 6),
    "Temperature_C": (-10, 55),
    "Humidity_pct": (0, 100),
    "Rainfall_mm": (0, 500),
    "PopulationIndex": (0, 100),
    "IndustrialIndex": (0, 100),
    "SolarGenerationIndex": (0, 1),
    "GridLoad_MW": (0, 5000),
}


def enforce_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Set physically-impossible numeric values to NaN (for later imputation)."""
    out = df.copy()
    for col, (lo, hi) in VALID_RANGES.items():
        if col in out.columns:
            bad = (out[col] < lo) | (out[col] > hi)
            out.loc[bad, col] = np.nan
    return out


def range_violation_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count out-of-range values per column (run on a type-coerced frame)."""
    rows = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            bad = ((df[col] < lo) | (df[col] > hi)).sum()
            rows[col] = {"valid_range": f"[{lo}, {hi}]", "out_of_range": int(bad)}
    return pd.DataFrame(rows).T.sort_values("out_of_range", ascending=False)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Force numeric columns to numbers; any non-parseable cell becomes NaN."""
    out = df.copy()
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip().str.title()
    return out


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing count and percentage, sorted worst-first."""
    n = len(df)
    miss = df.isna().sum()
    report = pd.DataFrame(
        {
            "missing_count": miss,
            "missing_pct": (miss / n * 100).round(2),
            "dtype": df.dtypes.astype(str),
        }
    )
    return report.sort_values("missing_pct", ascending=False)


def impute(
    df: pd.DataFrame,
    strategy: str = "median",
    knn_neighbors: int = 5,
) -> pd.DataFrame:
    """Impute missing values.

    Rows with a missing target are always dropped (can't learn or score them).
    Categorical columns are filled with the mode. Numeric columns are filled by
    ``strategy``: ``"median"`` (default, robust to outliers) or ``"knn"``.
    """
    out = df.copy()

    # Never impute the thing we're predicting.
    if TARGET in out.columns:
        out = out.dropna(subset=[TARGET]).reset_index(drop=True)

    for col in CATEGORICAL_COLS:
        if col in out.columns and out[col].isna().any():
            out[col] = out[col].fillna(out[col].mode(dropna=True).iloc[0])

    numeric_features = [c for c in NUMERIC_COLS if c in out.columns and c != TARGET]

    if strategy == "median":
        for col in numeric_features:
            out[col] = out[col].fillna(out[col].median())
    elif strategy == "mean":
        for col in numeric_features:
            out[col] = out[col].fillna(out[col].mean())
    elif strategy == "knn":
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=knn_neighbors)
        out[numeric_features] = imputer.fit_transform(out[numeric_features])
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy!r}")

    return out


def clean_data(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Full clean: coerce types then impute. Drops the Record_ID index column."""
    out = coerce_types(df)
    if "Record_ID" in out.columns:
        out = out.drop(columns=["Record_ID"])
    out = enforce_ranges(out)
    out = impute(out, strategy=strategy)
    return out
