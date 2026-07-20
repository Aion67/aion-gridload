"""Model training helpers.

A small registry of candidate regressors plus fit helpers, so notebooks can
screen many models with one loop and keep the comparison fair.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def candidate_models(random_state: int = 42) -> dict:
    """Return the shortlist of models for the basic screening pass.

    Linear models are wrapped in a StandardScaler pipeline; tree ensembles are
    scale-invariant so they run raw.
    """
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "LinearRegression": Pipeline(
            [("scale", StandardScaler()), ("model", LinearRegression())]
        ),
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }


def train_model(features, target, model=None, random_state: int = 42):
    """Train a single model (defaults to RandomForest) on the given data."""
    if model is None:
        model = RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        )
    model.fit(features, target)
    return model
