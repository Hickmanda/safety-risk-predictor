"""
Tests for threshold optimization and stochastic oracle utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.threshold_analysis import (
    compute_oracle_safe_probability,
    find_optimal_threshold,
    noise_pass_probability,
    predict_with_threshold,
    theoretical_bayes_accuracy_ceiling,
)


def test_predict_with_threshold() -> None:
    """
    Probabilities should be converted into SAFE/UNSAFE labels.
    """

    probabilities = np.array(
        [
            0.20,
            0.49,
            0.50,
            0.80,
        ]
    )

    predictions = predict_with_threshold(
        probabilities,
        0.50,
    )

    np.testing.assert_array_equal(
        predictions,
        np.array(
            [
                0,
                0,
                1,
                1,
            ]
        ),
    )


def test_threshold_optimization_can_improve_macro_f1() -> None:
    """
    Validation threshold search should find a better threshold
    when the default 0.50 threshold is suboptimal.
    """

    y_true = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    probability = np.array(
        [
            0.10,
            0.30,
            0.55,
            0.60,
            0.70,
            0.90,
        ]
    )

    threshold, results = (
        find_optimal_threshold(
            y_true,
            probability,
        )
    )

    best_row = (
        results.loc[
            results[
                "threshold"
            ]
            == threshold
        ]
        .iloc[0]
    )

    default_row = (
        results.iloc[
            (
                results[
                    "threshold"
                ]
                - 0.50
            )
            .abs()
            .argmin()
        ]
    )

    assert (
        best_row[
            "f1_macro"
        ]
        >= default_row[
            "f1_macro"
        ]
    )


def test_noise_probability_at_threshold_is_half() -> None:
    """
    A Gaussian centered exactly on the threshold should pass
    approximately half of the time.
    """

    probability = (
        noise_pass_probability(
            0.60
        )
    )

    assert float(
        probability
    ) == pytest.approx(
        0.5,
        abs=1e-10,
    )


def test_bayes_accuracy_ceiling() -> None:
    """
    Bayes accuracy ceiling should use max(p, 1-p) for each row.
    """

    probabilities = np.array(
        [
            0.0,
            0.5,
            1.0,
        ]
    )

    ceiling = (
        theoretical_bayes_accuracy_ceiling(
            probabilities
        )
    )

    assert ceiling == pytest.approx(
        (
            1.0
            + 0.5
            + 1.0
        )
        / 3.0
    )


def test_oracle_probability_is_bounded() -> None:
    """
    Oracle SAFE probabilities must always remain inside [0, 1].
    """

    dataframe = pd.DataFrame(
        {
            "SA": [
                0.50,
                0.70,
            ],
            "SK": [
                0.55,
                0.75,
            ],
            "SN": [
                0.60,
                0.70,
            ],
            "BA": [
                0.60,
                0.70,
            ],
            "PBC": [
                0.60,
                0.70,
            ],
            "reference_point": [
                0.60,
                0.60,
            ],
            "alpha": [
                0.88,
                0.88,
            ],
            "beta": [
                0.88,
                0.88,
            ],
            "lam": [
                1.18,
                1.18,
            ],
            "intention": [
                0.60,
                0.70,
            ],
            "day": [
                0,
                1,
            ],
        }
    )

    probabilities = (
        compute_oracle_safe_probability(
            dataframe
        )
    )

    assert len(
        probabilities
    ) == 2

    assert (
        probabilities >= 0.0
    ).all()

    assert (
        probabilities <= 1.0
    ).all()
