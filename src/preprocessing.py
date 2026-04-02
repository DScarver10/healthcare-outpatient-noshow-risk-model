"""Preprocessing utilities for tabular ML workflows."""

from collections.abc import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and return a copy."""
    return df.drop_duplicates().copy()


def cast_columns(df: pd.DataFrame, type_map: dict[str, str]) -> pd.DataFrame:
    """Cast selected columns to explicit dtypes when needed."""
    transformed = df.copy()
    for column, dtype in type_map.items():
        transformed[column] = transformed[column].astype(dtype)
    return transformed


def split_feature_types(
    df: pd.DataFrame, target_col: str, categorical_cols: Iterable[str] | None = None
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature lists, excluding the target."""
    feature_df = df.drop(columns=[target_col])

    if categorical_cols is None:
        categorical = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    else:
        categorical = list(categorical_cols)

    numeric = [col for col in feature_df.columns if col not in categorical]
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Create a reusable preprocessing block for sklearn pipelines."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
