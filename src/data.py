"""Data loading and saving helpers."""

from pathlib import Path

import pandas as pd


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a DataFrame to disk, creating parent folders when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def summarize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact column-level summary for inspection notebooks."""
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100).round(2),
            "n_unique": df.nunique(dropna=False),
        }
    )
    return summary.sort_values(["missing_pct", "n_unique"], ascending=[False, False])
