"""
Decision-threshold optimization and stochastic oracle analysis.

The trained classifier predicts P(safe). A default threshold of 0.50
favors the majority SAFE class and produces poor recall for UNSAFE
behavior.

This module:

1. Selects a decision threshold using the validation simulations only.
2. Compares threshold=0.50 with the optimized threshold.
3. Evaluates the selected threshold on the held-out test simulations.
4. Computes an oracle probability based on the known stochastic
   decision mechanism of the original ABM.
5. Estimates the theoretical Bayes accuracy ceiling caused by
   irreducible decision noise.

No test-set information is used to choose the threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

from src import params
from src.cpt import (
    compute_cpt_safe,
    compute_cpt_unsafe,
)
from src.ml_utils import (
    calculate_metrics,
    extract_xy,
    load_dataset,
    positive_class_probability,
    save_json,
    split_by_run,
)


# This value intentionally mirrors the Gaussian noise used by
# Worker.decide() in the original Project 2 implementation.
DECISION_NOISE_STD = 0.08

DEFAULT_THRESHOLD = 0.50


def predict_with_threshold(
    safe_probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert P(safe) into binary predictions.

    behavior=1 means SAFE.
    behavior=0 means UNSAFE.

    Raising the threshold makes the classifier more conservative:
    it requires greater confidence before declaring behavior safe.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = np.asarray(
        safe_probability,
        dtype=float,
    )

    return (
        probabilities >= threshold
    ).astype(int)


def find_optimal_threshold(
    y_true: pd.Series | np.ndarray,
    safe_probability: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 0.95,
    step: float = 0.005,
) -> tuple[
    float,
    pd.DataFrame,
]:
    """
    Find the probability threshold that maximizes macro F1.

    Threshold selection uses validation data only.

    Ties are resolved using balanced accuracy and then unsafe recall,
    because detecting unsafe behavior is safety-critical.
    """

    if step <= 0:
        raise ValueError(
            "step must be greater than zero."
        )

    thresholds = np.arange(
        minimum,
        maximum + step / 2.0,
        step,
    )

    rows: list[
        dict[str, float]
    ] = []

    for threshold in thresholds:
        threshold = float(
            round(
                threshold,
                6,
            )
        )

        predictions = predict_with_threshold(
            safe_probability,
            threshold,
        )

        rows.append(
            {
                "threshold": threshold,
                "accuracy": float(
                    accuracy_score(
                        y_true,
                        predictions,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        y_true,
                        predictions,
                    )
                ),
                "f1_macro": float(
                    f1_score(
                        y_true,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "recall_safe": float(
                    recall_score(
                        y_true,
                        predictions,
                        pos_label=1,
                        zero_division=0,
                    )
                ),
                "recall_unsafe": float(
                    recall_score(
                        y_true,
                        predictions,
                        pos_label=0,
                        zero_division=0,
                    )
                ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    ranked = results.sort_values(
        by=[
            "f1_macro",
            "balanced_accuracy",
            "recall_unsafe",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    best_threshold = float(
        ranked.iloc[0][
            "threshold"
        ]
    )

    return (
        best_threshold,
        results,
    )


def noise_pass_probability(
    mean_value: np.ndarray | float,
) -> np.ndarray:
    """
    Calculate P(noisy_value >= cognitive threshold).

    If:

        noisy_value ~ N(mean_value, 0.08)

    then:

        P(noisy_value >= threshold)
        = Phi((mean_value - threshold) / sigma)
    """

    values = np.asarray(
        mean_value,
        dtype=float,
    )

    standardized = (
        values
        - params.COGNITIVE_THRESHOLD
    ) / DECISION_NOISE_STD

    return ndtr(
        standardized
    )


def compute_oracle_safe_probability(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """
    Compute the exact conditional probability of SAFE behavior.

    The oracle knows the deterministic ABM equations and the
    distribution of the hidden Gaussian noise, but it does NOT know
    the random noise realization for each worker decision.

    Therefore this probability represents the best information
    available from the observable state variables.
    """

    reference_points = dataframe[
        "reference_point"
    ].to_numpy(
        dtype=float
    )

    alphas = dataframe[
        "alpha"
    ].to_numpy(
        dtype=float
    )

    betas = dataframe[
        "beta"
    ].to_numpy(
        dtype=float
    )

    lambdas = dataframe[
        "lam"
    ].to_numpy(
        dtype=float
    )

    row_count = len(
        dataframe
    )

    cpt_safe = np.fromiter(
        (
            compute_cpt_safe(
                reference_point=reference_point,
                alpha=alpha,
                beta=beta,
                lam=lam,
            )
            for (
                reference_point,
                alpha,
                beta,
                lam,
            ) in zip(
                reference_points,
                alphas,
                betas,
                lambdas,
            )
        ),
        dtype=float,
        count=row_count,
    )

    cpt_unsafe = np.fromiter(
        (
            compute_cpt_unsafe(
                reference_point=reference_point,
                alpha=alpha,
                beta=beta,
                lam=lam,
            )
            for (
                reference_point,
                alpha,
                beta,
                lam,
            ) in zip(
                reference_points,
                alphas,
                betas,
                lambdas,
            )
        ),
        dtype=float,
        count=row_count,
    )

    difference = (
        cpt_safe
        - cpt_unsafe
    )

    safe_behavior_score = (
        1.0
        / (
            1.0
            + np.exp(
                -params.SIGMOID_K
                * difference
            )
        )
    )

    # The CPT part of the decision rule is deterministic.
    cpt_gate = (
        safe_behavior_score
        >= params.COGNITIVE_THRESHOLD
    )

    probability_sa_passes = (
        noise_pass_probability(
            dataframe[
                "SA"
            ].to_numpy(
                dtype=float
            )
        )
    )

    probability_sk_passes = (
        noise_pass_probability(
            dataframe[
                "SK"
            ].to_numpy(
                dtype=float
            )
        )
    )

    # SA and SK noises are generated independently in Worker.decide().
    safe_probability = (
        cpt_gate.astype(float)
        * probability_sa_passes
        * probability_sk_passes
    )

    return safe_probability


def theoretical_bayes_accuracy_ceiling(
    oracle_probability: np.ndarray,
) -> float:
    """
    Estimate the maximum expected accuracy possible from observed state.

    For binary labels with known conditional probability p, the Bayes
    optimal accuracy for one observation is max(p, 1-p).

    Averaging this value estimates the accuracy ceiling imposed by
    stochastic label noise.
    """

    probabilities = np.asarray(
        oracle_probability,
        dtype=float,
    )

    if (
        probabilities.min() < 0.0
        or probabilities.max() > 1.0
    ):
        raise ValueError(
            "Oracle probabilities must be inside [0, 1]."
        )

    return float(
        np.maximum(
            probabilities,
            1.0 - probabilities,
        ).mean()
    )


def save_threshold_plot(
    results: pd.DataFrame,
    selected_threshold: float,
    path: Path,
) -> None:
    """
    Plot validation metrics as a function of decision threshold.
    """

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.plot(
        results[
            "threshold"
        ],
        results[
            "f1_macro"
        ],
        label="Macro F1",
    )

    axis.plot(
        results[
            "threshold"
        ],
        results[
            "recall_unsafe"
        ],
        label="Unsafe Recall",
    )

    axis.plot(
        results[
            "threshold"
        ],
        results[
            "recall_safe"
        ],
        label="Safe Recall",
    )

    axis.axvline(
        selected_threshold,
        linestyle="--",
        label=(
            "Selected threshold "
            f"({selected_threshold:.3f})"
        ),
    )

    axis.axvline(
        DEFAULT_THRESHOLD,
        linestyle=":",
        label="Default threshold (0.50)",
    )

    axis.set_xlabel(
        "P(safe) decision threshold"
    )

    axis.set_ylabel(
        "Metric"
    )

    axis.set_title(
        "Validation Threshold Optimization"
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def update_model_metadata(
    metadata_path: Path,
    threshold: float,
    validation_metrics: dict[
        str,
        float,
    ],
    test_metrics: dict[
        str,
        float,
    ],
) -> dict[str, Any]:
    """
    Store the production decision threshold in model metadata.
    """

    if metadata_path.exists():
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )
    else:
        metadata = {}

    metadata[
        "decision_threshold"
    ] = threshold

    metadata[
        "threshold_selection"
    ] = {
        "dataset": (
            "validation runs 80-89"
        ),
        "objective": (
            "macro F1"
        ),
        "default_threshold": (
            DEFAULT_THRESHOLD
        ),
        "selected_threshold": (
            threshold
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "test_metrics": (
            test_metrics
        ),
    }

    save_json(
        metadata,
        metadata_path,
    )

    return metadata


def run_threshold_analysis(
    data_path: Path,
    model_path: Path,
    metadata_path: Path,
    reports_dir: Path,
    tracking_uri: str,
) -> dict[str, Any]:
    """
    Run threshold optimization and stochastic oracle analysis.
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
        validation_dataframe,
        test_dataframe,
    ) = split_by_run(
        dataframe
    )

    (
        x_validation,
        y_validation,
    ) = extract_xy(
        validation_dataframe
    )

    (
        x_test,
        y_test,
    ) = extract_xy(
        test_dataframe
    )

    model = joblib.load(
        model_path
    )

    validation_probability = (
        positive_class_probability(
            model,
            x_validation,
        )
    )

    test_probability = (
        positive_class_probability(
            model,
            x_test,
        )
    )

    # ------------------------------------------------------------
    # Default 0.50 threshold
    # ------------------------------------------------------------

    validation_default_prediction = (
        predict_with_threshold(
            validation_probability,
            DEFAULT_THRESHOLD,
        )
    )

    validation_default_metrics = (
        calculate_metrics(
            y_validation,
            validation_default_prediction,
            validation_probability,
        )
    )

    # ------------------------------------------------------------
    # Optimize threshold using validation data only
    # ------------------------------------------------------------

    (
        selected_threshold,
        threshold_results,
    ) = find_optimal_threshold(
        y_validation,
        validation_probability,
    )

    validation_optimized_prediction = (
        predict_with_threshold(
            validation_probability,
            selected_threshold,
        )
    )

    validation_optimized_metrics = (
        calculate_metrics(
            y_validation,
            validation_optimized_prediction,
            validation_probability,
        )
    )

    # ------------------------------------------------------------
    # Apply the fixed selected threshold to held-out test data
    # ------------------------------------------------------------

    test_default_prediction = (
        predict_with_threshold(
            test_probability,
            DEFAULT_THRESHOLD,
        )
    )

    test_default_metrics = (
        calculate_metrics(
            y_test,
            test_default_prediction,
            test_probability,
        )
    )

    test_optimized_prediction = (
        predict_with_threshold(
            test_probability,
            selected_threshold,
        )
    )

    test_optimized_metrics = (
        calculate_metrics(
            y_test,
            test_optimized_prediction,
            test_probability,
        )
    )

    # ------------------------------------------------------------
    # Stochastic oracle
    # ------------------------------------------------------------

    print()
    print(
        "Computing stochastic oracle probabilities..."
    )

    oracle_test_probability = (
        compute_oracle_safe_probability(
            test_dataframe
        )
    )

    oracle_prediction = (
        predict_with_threshold(
            oracle_test_probability,
            DEFAULT_THRESHOLD,
        )
    )

    oracle_observed_metrics = (
        calculate_metrics(
            y_test,
            oracle_prediction,
            oracle_test_probability,
        )
    )

    bayes_accuracy_ceiling = (
        theoretical_bayes_accuracy_ceiling(
            oracle_test_probability
        )
    )

    # ------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------

    threshold_results.to_csv(
        reports_dir
        / "threshold_search.csv",
        index=False,
    )

    save_threshold_plot(
        threshold_results,
        selected_threshold,
        reports_dir
        / "threshold_optimization.png",
    )

    comparison = pd.DataFrame(
        [
            {
                "evaluation": (
                    "validation_default"
                ),
                "threshold": (
                    DEFAULT_THRESHOLD
                ),
                **validation_default_metrics,
            },
            {
                "evaluation": (
                    "validation_optimized"
                ),
                "threshold": (
                    selected_threshold
                ),
                **validation_optimized_metrics,
            },
            {
                "evaluation": (
                    "test_default"
                ),
                "threshold": (
                    DEFAULT_THRESHOLD
                ),
                **test_default_metrics,
            },
            {
                "evaluation": (
                    "test_optimized"
                ),
                "threshold": (
                    selected_threshold
                ),
                **test_optimized_metrics,
            },
            {
                "evaluation": (
                    "test_stochastic_oracle"
                ),
                "threshold": (
                    DEFAULT_THRESHOLD
                ),
                **oracle_observed_metrics,
            },
        ]
    )

    comparison.to_csv(
        reports_dir
        / "threshold_comparison.csv",
        index=False,
    )

    results: dict[str, Any] = {
        "selected_threshold": (
            selected_threshold
        ),
        "selection_objective": (
            "validation macro F1"
        ),
        "validation": {
            "default": (
                validation_default_metrics
            ),
            "optimized": (
                validation_optimized_metrics
            ),
        },
        "test": {
            "default": (
                test_default_metrics
            ),
            "optimized": (
                test_optimized_metrics
            ),
        },
        "stochastic_oracle": {
            "observed_test_metrics": (
                oracle_observed_metrics
            ),
            "theoretical_bayes_accuracy_ceiling": (
                bayes_accuracy_ceiling
            ),
            "noise_standard_deviation": (
                DECISION_NOISE_STD
            ),
        },
    }

    save_json(
        results,
        reports_dir
        / "threshold_analysis.json",
    )

    metadata = update_model_metadata(
        metadata_path=metadata_path,
        threshold=selected_threshold,
        validation_metrics=(
            validation_optimized_metrics
        ),
        test_metrics=(
            test_optimized_metrics
        ),
    )

    # ------------------------------------------------------------
    # Attach threshold analysis to the winning MLflow run
    # ------------------------------------------------------------

    mlflow_run_id = metadata.get(
        "mlflow_run_id"
    )

    if mlflow_run_id:
        mlflow.set_tracking_uri(
            tracking_uri
        )

        with mlflow.start_run(
            run_id=mlflow_run_id
        ):
            mlflow.log_param(
                "decision_threshold",
                selected_threshold,
            )

            mlflow.log_metrics(
                {
                    "threshold_val_f1_macro": (
                        validation_optimized_metrics[
                            "f1_macro"
                        ]
                    ),
                    "threshold_val_recall_unsafe": (
                        validation_optimized_metrics[
                            "recall_unsafe"
                        ]
                    ),
                    "threshold_test_f1_macro": (
                        test_optimized_metrics[
                            "f1_macro"
                        ]
                    ),
                    "threshold_test_recall_unsafe": (
                        test_optimized_metrics[
                            "recall_unsafe"
                        ]
                    ),
                    "oracle_test_roc_auc": (
                        oracle_observed_metrics[
                            "roc_auc"
                        ]
                    ),
                    "oracle_bayes_accuracy_ceiling": (
                        bayes_accuracy_ceiling
                    ),
                }
            )

            mlflow.log_artifact(
                str(
                    reports_dir
                    / "threshold_optimization.png"
                ),
                artifact_path=(
                    "threshold_analysis"
                ),
            )

            mlflow.log_artifact(
                str(
                    reports_dir
                    / "threshold_comparison.csv"
                ),
                artifact_path=(
                    "threshold_analysis"
                ),
            )

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "THRESHOLD OPTIMIZATION"
    )
    print("=" * 72)

    print(
        f"Default threshold      : "
        f"{DEFAULT_THRESHOLD:.3f}"
    )

    print(
        f"Selected threshold     : "
        f"{selected_threshold:.3f}"
    )

    print()
    print(
        "VALIDATION"
    )

    print(
        f"Macro F1 default       : "
        f"{validation_default_metrics['f1_macro']:.4f}"
    )

    print(
        f"Macro F1 optimized     : "
        f"{validation_optimized_metrics['f1_macro']:.4f}"
    )

    print(
        f"Unsafe recall default  : "
        f"{validation_default_metrics['recall_unsafe']:.4f}"
    )

    print(
        f"Unsafe recall optimized: "
        f"{validation_optimized_metrics['recall_unsafe']:.4f}"
    )

    print()
    print(
        "HELD-OUT TEST"
    )

    print(
        f"Macro F1 default       : "
        f"{test_default_metrics['f1_macro']:.4f}"
    )

    print(
        f"Macro F1 optimized     : "
        f"{test_optimized_metrics['f1_macro']:.4f}"
    )

    print(
        f"Unsafe recall default  : "
        f"{test_default_metrics['recall_unsafe']:.4f}"
    )

    print(
        f"Unsafe recall optimized: "
        f"{test_optimized_metrics['recall_unsafe']:.4f}"
    )

    print()
    print(
        "STOCHASTIC ORACLE"
    )

    print(
        f"Oracle ROC-AUC         : "
        f"{oracle_observed_metrics['roc_auc']:.4f}"
    )

    print(
        f"Observed oracle accuracy: "
        f"{oracle_observed_metrics['accuracy']:.4f}"
    )

    print(
        f"Bayes accuracy ceiling : "
        f"{bayes_accuracy_ceiling:.4f}"
    )

    print("=" * 72)

    return results


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Optimize the production decision threshold "
            "and estimate stochastic prediction limits."
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
        "--metadata",
        type=Path,
        default=Path(
            "models/model_metadata.json"
        ),
    )

    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(
            "reports"
        ),
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=(
            "sqlite:///mlflow.db"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_arguments()

    run_threshold_analysis(
        data_path=args.data,
        model_path=args.model,
        metadata_path=args.metadata,
        reports_dir=args.reports_dir,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
