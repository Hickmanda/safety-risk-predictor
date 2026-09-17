"""
Evaluate the final selected safety classifier.

The script loads the untouched test simulations, calculates final
metrics and generates portfolio-ready evaluation artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline

from src.generate_data import (
    FEATURE_COLUMNS,
)
from src.ml_utils import (
    calculate_metrics,
    extract_xy,
    load_dataset,
    positive_class_probability,
    save_json,
    split_by_run,
)


def extract_feature_importance(
    model: Any,
) -> np.ndarray | None:
    """
    Extract feature importance from supported model types.

    Logistic Regression uses the absolute model coefficients.
    Tree models expose feature_importances_ directly.
    """

    estimator = model

    if isinstance(
        model,
        Pipeline,
    ):
        estimator = (
            model.named_steps.get(
                "classifier"
            )
        )

    if estimator is None:
        return None

    if hasattr(
        estimator,
        "feature_importances_",
    ):
        return np.asarray(
            estimator.feature_importances_,
            dtype=float,
        )

    if hasattr(
        estimator,
        "coef_",
    ):
        coefficients = np.asarray(
            estimator.coef_,
            dtype=float,
        )

        if coefficients.ndim == 2:
            coefficients = (
                coefficients[0]
            )

        return np.abs(
            coefficients
        )

    return None


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: Path,
) -> None:
    """
    Save a confusion matrix figure.
    """

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[
            "Unsafe",
            "Safe",
        ],
        ax=axis,
    )

    axis.set_title(
        "Confusion Matrix - Test Set"
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_roc_curve(
    y_true: pd.Series,
    y_probability: np.ndarray,
    path: Path,
) -> None:
    """
    Save the ROC curve for the final test set.
    """

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    RocCurveDisplay.from_predictions(
        y_true,
        y_probability,
        ax=axis,
    )

    axis.set_title(
        "ROC Curve - Test Set"
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_feature_importance(
    model: Any,
    path: Path,
) -> bool:
    """
    Save feature importance if the model exposes it.

    Returns True when a plot was created.
    """

    importance = (
        extract_feature_importance(
            model
        )
    )

    if importance is None:
        return False

    if len(importance) != len(
        FEATURE_COLUMNS
    ):
        return False

    importance_frame = pd.DataFrame(
        {
            "feature": (
                FEATURE_COLUMNS
            ),
            "importance": (
                importance
            ),
        }
    )

    importance_frame = (
        importance_frame.sort_values(
            "importance",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.barh(
        importance_frame[
            "feature"
        ],
        importance_frame[
            "importance"
        ],
    )

    axis.set_xlabel(
        "Importance"
    )

    axis.set_title(
        "Feature Importance"
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return True


def evaluate_model(
    data_path: Path,
    model_path: Path,
    reports_dir: Path,
) -> dict[str, float]:
    """
    Evaluate the saved best model on runs 90-99.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: "
            f"{model_path}"
        )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_dataset(
        data_path
    )

    (
        _,
        _,
        test_dataframe,
    ) = split_by_run(
        dataframe
    )

    x_test, y_test = (
        extract_xy(
            test_dataframe
        )
    )

    model = joblib.load(
        model_path
    )

    predictions = (
        model.predict(
            x_test
        )
    )

    probabilities = (
        positive_class_probability(
            model,
            x_test,
        )
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    save_json(
        metrics,
        reports_dir
        / "evaluation_metrics.json",
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "unsafe",
            "safe",
        ],
        output_dict=True,
        zero_division=0,
    )

    save_json(
        report,
        reports_dir
        / "classification_report.json",
    )

    save_confusion_matrix(
        y_test,
        predictions,
        reports_dir
        / "confusion_matrix.png",
    )

    save_roc_curve(
        y_test,
        probabilities,
        reports_dir
        / "roc_curve.png",
    )

    created_importance = (
        save_feature_importance(
            model,
            reports_dir
            / "feature_importance.png",
        )
    )

    print("=" * 72)
    print(
        "FINAL MODEL EVALUATION"
    )
    print("=" * 72)

    for (
        metric_name,
        metric_value,
    ) in metrics.items():
        print(
            f"{metric_name:<20}: "
            f"{metric_value:.4f}"
        )

    print("=" * 72)

    print(
        "Saved: "
        "reports/evaluation_metrics.json"
    )

    print(
        "Saved: "
        "reports/classification_report.json"
    )

    print(
        "Saved: "
        "reports/confusion_matrix.png"
    )

    print(
        "Saved: "
        "reports/roc_curve.png"
    )

    if created_importance:
        print(
            "Saved: "
            "reports/feature_importance.png"
        )

    return metrics


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the selected "
            "construction safety classifier."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/safety_dataset.csv"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/best_model.pkl"
        ),
    )

    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(
            "reports"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_arguments()

    evaluate_model(
        data_path=args.data,
        model_path=args.model,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
