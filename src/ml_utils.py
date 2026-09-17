"""
Shared utilities for machine-learning training and evaluation.

The module centralizes dataset validation, leakage-safe splitting,
feature extraction and classification metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.generate_data import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)


TRAIN_RUNS = range(0, 80)
VALIDATION_RUNS = range(80, 90)
TEST_RUNS = range(90, 100)


REQUIRED_COLUMNS = (
    METADATA_COLUMNS
    + FEATURE_COLUMNS
    + [TARGET_COLUMN]
)


def load_dataset(
    path: Path,
) -> pd.DataFrame:
    """
    Load and validate the generated ABM dataset.

    Dataset validation is performed before model training so that
    schema errors cannot silently propagate into ML experiments.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    dataframe = pd.read_csv(path)

    missing_columns = (
        set(REQUIRED_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if (
        dataframe[
            REQUIRED_COLUMNS
        ]
        .isnull()
        .any()
        .any()
    ):
        raise ValueError(
            "Dataset contains missing values."
        )

    target_values = set(
        dataframe[
            TARGET_COLUMN
        ].unique()
    )

    if not target_values.issubset(
        {0, 1}
    ):
        raise ValueError(
            "Target column must contain only 0 and 1."
        )

    return dataframe


def split_by_run(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split the dataset by independent simulation run.

    Random row splitting would leak correlated worker trajectories
    between training and evaluation sets. Splitting by run_id keeps
    every complete ABM simulation inside exactly one dataset split.
    """

    train = dataframe[
        dataframe["run_id"].isin(
            TRAIN_RUNS
        )
    ].copy()

    validation = dataframe[
        dataframe["run_id"].isin(
            VALIDATION_RUNS
        )
    ].copy()

    test = dataframe[
        dataframe["run_id"].isin(
            TEST_RUNS
        )
    ].copy()

    if train.empty:
        raise ValueError(
            "Training split is empty."
        )

    if validation.empty:
        raise ValueError(
            "Validation split is empty."
        )

    if test.empty:
        raise ValueError(
            "Test split is empty."
        )

    train_runs = set(
        train["run_id"].unique()
    )

    validation_runs = set(
        validation["run_id"].unique()
    )

    test_runs = set(
        test["run_id"].unique()
    )

    if train_runs & validation_runs:
        raise RuntimeError(
            "Train and validation run IDs overlap."
        )

    if train_runs & test_runs:
        raise RuntimeError(
            "Train and test run IDs overlap."
        )

    if validation_runs & test_runs:
        raise RuntimeError(
            "Validation and test run IDs overlap."
        )

    return (
        train,
        validation,
        test,
    )


def extract_xy(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Extract predictive features and target.

    Metadata fields such as run_id, seed and worker_id are deliberately
    excluded so the ML model cannot memorize simulation identities.
    """

    features = dataframe[
        FEATURE_COLUMNS
    ].copy()

    target = dataframe[
        TARGET_COLUMN
    ].astype(int)

    return (
        features,
        target,
    )


def positive_class_probability(
    model: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Return the predicted probability for behavior=1.

    All models used in this project expose predict_proba(), but this
    helper keeps probability extraction in one place.
    """

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError(
            "Model does not support predict_proba()."
        )

    probabilities = model.predict_proba(
        features
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "predict_proba() returned an invalid shape."
        )

    if probabilities.shape[1] != 2:
        raise ValueError(
            "Binary classifier must return two probability columns."
        )

    return probabilities[:, 1]


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """
    Calculate classification metrics.

    behavior=1 means SAFE and behavior=0 means UNSAFE.

    We report metrics for both classes because unsafe behavior is the
    safety-critical class, while safe behavior is the majority class.

    Macro F1 is used later for model selection because it gives equal
    importance to safe and unsafe behavior.
    """

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision_safe": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_safe": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_safe": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "precision_unsafe": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "recall_unsafe": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_unsafe": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        ),
    }


def class_distribution(
    target: pd.Series,
) -> dict[str, float | int]:
    """
    Return class counts and proportions for logging.
    """

    total = len(target)

    safe_count = int(
        (target == 1).sum()
    )

    unsafe_count = int(
        (target == 0).sum()
    )

    return {
        "total": total,
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "safe_rate": (
            safe_count / total
        ),
        "unsafe_rate": (
            unsafe_count / total
        ),
    }


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """
    Save structured metadata as human-readable JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )
