"""Data cleaning and preprocessing helpers.

The raw dataset has two kinds of dirt, seeded per column:
  1. Placeholder tokens instead of a real value, e.g. "N/A", "unknown", "###".
  2. Parseable-but-invalid values, e.g. Hour=30 (valid range is 0-23),
     Humidity_pct=-15 (valid range is 0-100).

Both audit_missing_values() (Task 1: find and quantify) and clean_data()
(Task 2/3: fix and evaluate) are built around that split so the same column
rules are used to report the problem and to fix it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Tokens that mean "no value" regardless of column, compared case-insensitively
# after stripping whitespace.
MISSING_TOKENS = {"", "n/a", "na", "unknown", "###", "null", "none", "-", "nan"}

# Expected numeric columns and the range of values that are physically
# plausible for each. Anything outside the range is treated as invalid data
# (not a real reading) and is set to missing rather than kept as-is.
NUMERIC_COLUMN_RANGES: dict[str, tuple[float | None, float | None]] = {
    "Hour": (0, 23),
    "DayOfWeek": (0, 6),
    "Temperature_C": (-10, 50),
    "Humidity_pct": (0, 100),
    "Rainfall_mm": (0, None),
    "PopulationIndex": (0, 100),
    "IndustrialIndex": (0, 100),
    "SolarGenerationIndex": (0, None),
    "GridLoad_MW": (0, None),
}

# Matches an optional sign, digits, and an optional decimal part anywhere in
# a string, e.g. pulls "900" out of "MW900". Used only as a fallback once a
# direct float() parse has already failed.
_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


def _coerce_numeric(series: pd.Series, value_range: tuple[float | None, float | None]) -> pd.Series:
    """Convert a raw (object-dtype) column to numeric, turning dirt into NaN.

    Order of operations matters: known placeholder tokens are cleared first,
    then a direct numeric parse is tried, then a regex fallback recovers
    values like "MW900" that carry a real number inside a non-numeric
    string. Finally, values outside the physically plausible range are
    treated as invalid and cleared too.
    """
    cleaned = series.astype(str).str.strip()
    is_placeholder = cleaned.str.lower().isin(MISSING_TOKENS)
    cleaned = cleaned.mask(is_placeholder)

    numeric = pd.to_numeric(cleaned, errors="coerce")

    # Fallback for strings that failed a direct parse but contain a number,
    # e.g. "MW900" -> 900. Only touches rows still NaN after the direct
    # parse and that had non-missing text to begin with.
    needs_fallback = numeric.isna() & cleaned.notna()
    for idx in cleaned[needs_fallback].index:
        match = _NUMBER_PATTERN.search(cleaned[idx])
        if match:
            numeric[idx] = float(match.group())

    low, high = value_range
    if low is not None:
        numeric = numeric.mask(numeric < low)
    if high is not None:
        numeric = numeric.mask(numeric > high)

    return numeric


def audit_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Report, per column, how much data is missing or invalid (Task 1).

    For the numeric columns in NUMERIC_COLUMN_RANGES this counts both
    placeholder tokens ("N/A", "###", ...) and out-of-range values as
    missing, since both mean "we don't have a usable reading" for that
    column. Non-numeric columns just report native NaNs.
    """
    rows = []
    for col in df.columns:
        if col in NUMERIC_COLUMN_RANGES:
            coerced = _coerce_numeric(df[col], NUMERIC_COLUMN_RANGES[col])
            missing = coerced.isna().sum()
        else:
            missing = df[col].isna().sum()
        rows.append(
            {
                "column": col,
                "missing_count": int(missing),
                "missing_pct": round(100 * missing / len(df), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False, ignore_index=True)


def clean_data(df: pd.DataFrame, impute_strategy: str = "median") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the raw dataframe and return (cleaned_df, report_df) (Task 2/3).

    Steps:
      1. Drop exact duplicate rows (same Record_ID and every other value repeated).
      2. Coerce each expected-numeric column, turning placeholder tokens and
         out-of-range values into NaN (see _coerce_numeric).
      3. Impute remaining NaNs per column:
         - "median" (default): robust to the outliers/skew typical of
           weather and load data, and cheap to justify/reproduce.
         - "knn": uses the other numeric columns to estimate a value;
           worth comparing against median in an imputation_* experiment
           notebook to see which one downstream models prefer.
      4. Tidy the one categorical column (Region) by stripping whitespace.

    The report_df returned alongside the cleaned data records missing counts
    before and after cleaning per column, so the notebook can show the effect
    of the imputation step (Task 3).
    """
    if impute_strategy not in {"median", "knn"}:
        raise ValueError('impute_strategy must be "median" or "knn"')

    cleaned = df.copy()
    n_before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    n_duplicates_dropped = n_before - len(cleaned)

    numeric_cols = [c for c in NUMERIC_COLUMN_RANGES if c in cleaned.columns]
    missing_before = {}
    for col in numeric_cols:
        cleaned[col] = _coerce_numeric(cleaned[col], NUMERIC_COLUMN_RANGES[col])
        missing_before[col] = int(cleaned[col].isna().sum())

    if impute_strategy == "median":
        for col in numeric_cols:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    else:
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=5)
        cleaned[numeric_cols] = imputer.fit_transform(cleaned[numeric_cols])

    if "Region" in cleaned.columns:
        cleaned["Region"] = cleaned["Region"].astype(str).str.strip()

    report = pd.DataFrame(
        {
            "column": numeric_cols,
            "missing_before": [missing_before[c] for c in numeric_cols],
            "missing_after": [int(cleaned[c].isna().sum()) for c in numeric_cols],
        }
    )
    report.attrs["duplicates_dropped"] = n_duplicates_dropped
    report.attrs["impute_strategy"] = impute_strategy
    return cleaned, report
