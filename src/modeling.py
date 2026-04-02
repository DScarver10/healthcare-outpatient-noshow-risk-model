"""Model training and validation helpers."""

from collections.abc import Iterable

import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline


def make_pipeline(preprocessor, estimator) -> Pipeline:
    """Combine preprocessing and estimator into one reproducible pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    stratify: pd.Series | None = None,
):
    """Create a reproducible train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def run_cross_validation(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: Iterable[str] | str,
    cv,
) -> pd.DataFrame:
    """Run cross-validation and return results as a tidy DataFrame."""
    results = cross_validate(
        model,
        X_train,
        y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        return_train_score=False,
    )
    return pd.DataFrame(results)


def fit_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit and return a cloned model to avoid mutating shared objects."""
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)
    return fitted_model


def save_model(model: Pipeline, path) -> None:
    """Persist a trained model artifact to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
