"""Feature engineering helpers.

Turns a cleaned dataframe into a model-ready numeric matrix. Because the data
is cross-sectional (Hour / DayOfWeek columns, no continuous timestamp), the
engineered features are calendar-cyclical, categorical encodings and simple
interactions rather than lag/rolling series features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loading import TARGET


def add_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Hour (period 24) and DayOfWeek (period 7) as sin/cos pairs.

    Cyclical encoding keeps hour 23 numerically close to hour 0, which a raw
    integer column cannot express.
    """
    out = df.copy()
    if "Hour" in out.columns:
        out["Hour_sin"] = np.sin(2 * np.pi * out["Hour"] / 24)
        out["Hour_cos"] = np.cos(2 * np.pi * out["Hour"] / 24)
    if "DayOfWeek" in out.columns:
        out["DoW_sin"] = np.sin(2 * np.pi * out["DayOfWeek"] / 7)
        out["DoW_cos"] = np.cos(2 * np.pi * out["DayOfWeek"] / 7)
    return out


def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable calendar flags derived from Hour / DayOfWeek."""
    out = df.copy()
    if "DayOfWeek" in out.columns:
        out["is_weekend"] = (out["DayOfWeek"] >= 5).astype(int)
    if "Hour" in out.columns:
        # Peak demand windows in typical daily load curves.
        out["is_daytime"] = out["Hour"].between(6, 18).astype(int)
        out["is_evening_peak"] = out["Hour"].between(18, 22).astype(int)
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cross terms that plausibly drive load (cooling demand, activity)."""
    out = df.copy()
    if {"Temperature_C", "Humidity_pct"}.issubset(out.columns):
        out["Temp_x_Humidity"] = out["Temperature_C"] * out["Humidity_pct"]
    if {"PopulationIndex", "IndustrialIndex"}.issubset(out.columns):
        out["Pop_x_Industrial"] = out["PopulationIndex"] * out["IndustrialIndex"]
    return out


def build_features(
    df: pd.DataFrame,
    cyclical: bool = True,
    calendar: bool = True,
    interactions: bool = True,
    one_hot: bool = True,
) -> pd.DataFrame:
    """Create the model-ready feature frame (still including the target column).

    Toggles let experiment notebooks measure each feature group's contribution.
    Region is one-hot encoded; original Region/Hour/DayOfWeek raw columns are
    kept alongside engineered ones (tree models tolerate the redundancy).
    """
    out = df.copy()
    if cyclical:
        out = add_cyclical(out)
    if calendar:
        out = add_calendar_flags(out)
    if interactions:
        out = add_interactions(out)
    if one_hot and "Region" in out.columns:
        dummies = pd.get_dummies(out["Region"], prefix="Region").astype(int)
        out = pd.concat([out.drop(columns=["Region"]), dummies], axis=1)
    return out


def split_xy(df: pd.DataFrame):
    """Split a feature frame into (X numeric matrix, y target series)."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    # Keep only numeric columns (drop any leftover string cols defensively).
    X = X.select_dtypes(include=[np.number])
    return X, y
