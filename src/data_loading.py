"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a project CSV dataset from disk without altering its contents.

    Every column is read as-is (no dtype coercion, no missing-value handling)
    so the raw dirty values (e.g. "N/A", "unknown", "###") stay visible for
    the data-understanding audit. Cleaning belongs in
    ``src.preprocessing.clean_data``, not here.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)
