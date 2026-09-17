"""
Tests for machine-learning utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.generate_data import (
    FEATURE_COLUMNS,
)
from src.ml_utils import (
    calculate_metrics,
    extract_xy,
    split_by_run,
)


def make_test_dataframe() -> pd.DataFrame:
    """
    Create a minimal dataset containing all 100 simulation run IDs.
    """

    rows = []

    for run_id in range(
        100
    ):
        row = {
            "run_id": run_id,
            "seed": run_id,
            "worker_id": 0,
            "SA": 0.6,
            "SK": 0.7,
            "SN": 0.6,
            "BA": 0.7,
            "PBC": 0.6,
            "reference_point": 0.6,
            "alpha": 0.88,
            "beta": 0.88,
            "lam": 1.18,
            "intention": (
                0.633333
            ),
            "day": run_id,
            "behavior": (
                run_id % 2
            ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def test_run_based_split_sizes() -> None:
    """
    The fixed split should produce 80/10/10 simulation runs.
    """

    dataframe = (
        make_test_dataframe()
    )

    (
        train,
        validation,
        test,
    ) = split_by_run(
        dataframe
    )

    assert len(train) == 80
    assert len(validation) == 10
    assert len(test) == 10


def test_run_splits_do_not_overlap() -> None:
    """
    No simulation run should appear in more than one split.
    """

    dataframe = (
        make_test_dataframe()
    )

    (
        train,
        validation,
        test,
    ) = split_by_run(
        dataframe
    )

    train_runs = set(
        train["run_id"]
    )

    validation_runs = set(
        validation["run_id"]
    )

    test_runs = set(
        test["run_id"]
    )

    assert not (
        train_runs
        & validation_runs
    )

    assert not (
        train_runs
        & test_runs
    )

    assert not (
        validation_runs
        & test_runs
    )


def test_metadata_is_not_a_feature() -> None:
    """
    Simulation identifiers must not be exposed to the ML model.
    """

    dataframe = (
        make_test_dataframe()
    )

    features, _ = extract_xy(
        dataframe
    )

    assert list(
        features.columns
    ) == FEATURE_COLUMNS

    assert (
        "run_id"
        not in features.columns
    )

    assert (
        "seed"
        not in features.columns
    )

    assert (
        "worker_id"
        not in features.columns
    )


def test_metrics_perfect_prediction() -> None:
    """
    Perfect predictions should produce perfect classification metrics.
    """

    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    y_pred = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    y_probability = np.array(
        [
            0.05,
            0.10,
            0.90,
            0.95,
        ]
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_probability,
    )

    assert metrics[
        "accuracy"
    ] == pytest.approx(
        1.0
    )

    assert metrics[
        "f1_macro"
    ] == pytest.approx(
        1.0
    )

    assert metrics[
        "roc_auc"
    ] == pytest.approx(
        1.0
    )


def test_metrics_include_unsafe_recall() -> None:
    """
    Safety evaluation should explicitly report unsafe-behavior recall.
    """

    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    y_pred = np.array(
        [
            0,
            1,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.6,
            0.8,
            0.9,
        ]
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        probabilities,
    )

    assert (
        "recall_unsafe"
        in metrics
    )

    assert metrics[
        "recall_unsafe"
    ] == pytest.approx(
        0.5
    )
