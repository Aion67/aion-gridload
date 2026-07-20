"""Dataset loading helpers for the Uganda grid-load project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repository root = two levels up from this file (src/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = REPO_ROOT / "data" / "raw" / "Uganda_Grid_Load_Dataset.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

# Tokens that appear in the raw CSV in place of real numbers. They must be
# treated as missing so numeric columns parse cleanly.
JUNK_TOKENS = ["N/A", "NA", "n/a", "unknown", "Unknown", "###", "?", "-", ""]

TARGET = "GridLoad_MW"
CATEGORICAL_COLS = ["Region"]
# Numeric columns as they should end up after cleaning.
NUMERIC_COLS = [
    "Hour",
    "DayOfWeek",
    "Temperature_C",
    "Humidity_pct",
    "Rainfall_mm",
    "PopulationIndex",
    "IndustrialIndex",
    "SolarGenerationIndex",
    "GridLoad_MW",
]


def load_dataset(path: str | Path = RAW_DATA) -> pd.DataFrame:
    """Load the raw dataset, mapping known junk tokens to NaN.

    The raw CSV mixes strings like ``"hot"``, ``"###"`` and ``"unknown"`` into
    otherwise-numeric columns. We load everything as-is (na_values maps the
    obvious junk to NaN); numeric coercion happens in preprocessing so the
    "before" state is still inspectable.
    """
    path = Path(path)
    df = pd.read_csv(path, na_values=JUNK_TOKENS, keep_default_na=True)
    return df
