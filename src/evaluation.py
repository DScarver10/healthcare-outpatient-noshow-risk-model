"""Evaluation helpers for classification and regression projects."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_report_dict(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Return core classification metrics in a simple dictionary."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)

    return metrics


def regression_report_dict(y_true, y_pred) -> dict[str, float]:
    """Return core regression metrics in a simple dictionary."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred) -> None:
    """Plot a confusion matrix for quick error inspection."""
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()


def plot_regression_residuals(y_true, y_pred) -> None:
    """Plot residuals to check for large patterns left unexplained."""
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()


def metrics_frame(metrics_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert a nested metrics dictionary into a comparison table."""
    return pd.DataFrame(metrics_by_model).T.sort_index()


def classification_thresholds_frame(y_true, y_score, thresholds: list[float]) -> pd.DataFrame:
    """Summarize threshold trade-offs for a binary classifier."""
    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)
