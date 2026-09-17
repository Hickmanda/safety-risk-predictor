"""
Train and compare construction safety classifiers.

Models:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost

Experiments are tracked with MLflow.

Model selection is performed using validation macro F1.
The untouched test set is evaluated only after the best model
has been selected.
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)
from xgboost import XGBClassifier

from src.generate_data import (
    FEATURE_COLUMNS,
)
from src.ml_utils import (
    calculate_metrics,
    class_distribution,
    extract_xy,
    load_dataset,
    positive_class_probability,
    save_json,
    split_by_run,
)


RANDOM_STATE = 42

SELECTION_METRIC = "f1_macro"


def build_models() -> dict[
    str,
    Any,
]:
    """
    Build the three candidate classifiers.

    Hyperparameters are intentionally conservative so that training
    remains practical on a CPU while still providing meaningful model
    comparison.
    """

    logistic_regression = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = (
        RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )

    xgboost = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return {
        "logistic_regression": (
            logistic_regression
        ),
        "random_forest": (
            random_forest
        ),
        "xgboost": xgboost,
    }


def model_parameters(
    model_name: str,
    model: Any,
) -> dict[str, Any]:
    """
    Return a compact set of meaningful model parameters for MLflow.
    """

    if model_name == "logistic_regression":
        classifier = (
            model.named_steps[
                "classifier"
            ]
        )

        return {
            "model_type": (
                "LogisticRegression"
            ),
            "max_iter": (
                classifier.max_iter
            ),
            "random_state": (
                RANDOM_STATE
            ),
            "scaler": (
                "StandardScaler"
            ),
        }

    if model_name == "random_forest":
        return {
            "model_type": (
                "RandomForestClassifier"
            ),
            "n_estimators": (
                model.n_estimators
            ),
            "max_depth": (
                model.max_depth
            ),
            "min_samples_leaf": (
                model.min_samples_leaf
            ),
            "max_features": (
                model.max_features
            ),
            "random_state": (
                RANDOM_STATE
            ),
        }

    if model_name == "xgboost":
        return {
            "model_type": (
                "XGBClassifier"
            ),
            "n_estimators": (
                model.n_estimators
            ),
            "max_depth": (
                model.max_depth
            ),
            "learning_rate": (
                model.learning_rate
            ),
            "subsample": (
                model.subsample
            ),
            "colsample_bytree": (
                model.colsample_bytree
            ),
            "tree_method": (
                model.tree_method
            ),
            "random_state": (
                RANDOM_STATE
            ),
        }

    return {
        "model_type": (
            type(model).__name__
        )
    }


def log_mlflow_model(
    model_name: str,
    model: Any,
    x_example: pd.DataFrame,
) -> None:
    """
    Log a trained model using the appropriate MLflow flavor.

    MLflow 3 uses the pickle-free skops format for scikit-learn
    models by default. RandomForestClassifier contains the internal
    sklearn.tree._tree.Tree type, so we explicitly trust only that
    known scikit-learn type.
    """

    prediction_example = model.predict(
        x_example
    )

    signature = infer_signature(
        x_example,
        prediction_example,
    )

    if model_name == "xgboost":
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            input_example=x_example,
        )

        return

    if model_name == "random_forest":
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="skops",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
            ],
            signature=signature,
            input_example=x_example,
        )

        return

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        serialization_format="skops",
        signature=signature,
        input_example=x_example,
    )


def train_models(
    data_path: Path,
    models_dir: Path,
    reports_dir: Path,
    tracking_uri: str,
    experiment_name: str,
) -> dict[str, Any]:
    """
    Train all candidate models and save the best one.

    Selection is based only on validation macro F1.
    Test metrics are calculated once for the winning model.
    """

    print("=" * 72)
    print(
        "CONSTRUCTION SAFETY ML TRAINING"
    )
    print("=" * 72)

    dataframe = load_dataset(
        data_path
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = split_by_run(
        dataframe
    )

    x_train, y_train = (
        extract_xy(
            train_df
        )
    )

    x_validation, y_validation = (
        extract_xy(
            validation_df
        )
    )

    x_test, y_test = (
        extract_xy(
            test_df
        )
    )

    print(
        f"Dataset rows      : "
        f"{len(dataframe):,}"
    )

    print(
        f"Training rows     : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows   : "
        f"{len(validation_df):,}"
    )

    print(
        f"Test rows         : "
        f"{len(test_df):,}"
    )

    train_distribution = (
        class_distribution(
            y_train
        )
    )

    print(
        "Training balance  : "
        f"{train_distribution['safe_rate']:.2%} safe / "
        f"{train_distribution['unsafe_rate']:.2%} unsafe"
    )

    print("=" * 72)

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        experiment_name
    )

    candidate_models = (
        build_models()
    )

    comparison_rows: list[
        dict[str, Any]
    ] = []

    best_model: Any | None = None
    best_model_name = ""
    best_score = -1.0
    best_run_id = ""
    best_validation_metrics: dict[
        str,
        float,
    ] = {}

    for (
        model_name,
        model,
    ) in candidate_models.items():
        print()
        print("-" * 72)

        print(
            f"Training: {model_name}"
        )

        print("-" * 72)

        with mlflow.start_run(
            run_name=model_name
        ) as run:
            parameters = (
                model_parameters(
                    model_name,
                    model,
                )
            )

            parameters.update(
                {
                    "train_rows": (
                        len(x_train)
                    ),
                    "validation_rows": (
                        len(
                            x_validation
                        )
                    ),
                    "feature_count": (
                        len(
                            FEATURE_COLUMNS
                        )
                    ),
                    "selection_metric": (
                        SELECTION_METRIC
                    ),
                }
            )

            mlflow.log_params(
                parameters
            )

            start_time = (
                time.perf_counter()
            )

            model.fit(
                x_train,
                y_train,
            )

            training_seconds = (
                time.perf_counter()
                - start_time
            )

            validation_predictions = (
                model.predict(
                    x_validation
                )
            )

            validation_probabilities = (
                positive_class_probability(
                    model,
                    x_validation,
                )
            )

            validation_metrics = (
                calculate_metrics(
                    y_validation,
                    validation_predictions,
                    validation_probabilities,
                )
            )

            mlflow_metrics = {
                (
                    f"val_{name}"
                ): value
                for (
                    name,
                    value,
                ) in (
                    validation_metrics.items()
                )
            }

            mlflow_metrics[
                "training_seconds"
            ] = training_seconds

            mlflow.log_metrics(
                mlflow_metrics
            )

            mlflow.set_tags(
                {
                    "project": (
                        "construction-safety-risk-predictor"
                    ),
                    "dataset_source": (
                        "agent-based-simulation"
                    ),
                    "split_strategy": (
                        "simulation-run-based"
                    ),
                }
            )

            input_example = (
                x_train.head(5)
            )

            log_mlflow_model(
                model_name,
                model,
                input_example,
            )

            result_row = {
                "model": model_name,
                "training_seconds": (
                    training_seconds
                ),
                **validation_metrics,
                "mlflow_run_id": (
                    run.info.run_id
                ),
            }

            comparison_rows.append(
                result_row
            )

            score = (
                validation_metrics[
                    SELECTION_METRIC
                ]
            )

            print(
                f"Training time : "
                f"{training_seconds:.2f} s"
            )

            print(
                f"Accuracy      : "
                f"{validation_metrics['accuracy']:.4f}"
            )

            print(
                f"Macro F1      : "
                f"{validation_metrics['f1_macro']:.4f}"
            )

            print(
                f"Safe F1       : "
                f"{validation_metrics['f1_safe']:.4f}"
            )

            print(
                f"Unsafe F1     : "
                f"{validation_metrics['f1_unsafe']:.4f}"
            )

            print(
                f"Unsafe recall : "
                f"{validation_metrics['recall_unsafe']:.4f}"
            )

            print(
                f"ROC-AUC       : "
                f"{validation_metrics['roc_auc']:.4f}"
            )

            if score > best_score:
                best_score = score

                best_model = model

                best_model_name = (
                    model_name
                )

                best_run_id = (
                    run.info.run_id
                )

                best_validation_metrics = (
                    validation_metrics.copy()
                )

            else:
                del model
                gc.collect()

    if best_model is None:
        raise RuntimeError(
            "No model was successfully trained."
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = (
        comparison.sort_values(
            SELECTION_METRIC,
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    comparison_path = (
        reports_dir
        / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print()
    print("=" * 72)
    print(
        "BEST VALIDATION MODEL"
    )
    print("=" * 72)

    print(
        f"Model          : "
        f"{best_model_name}"
    )

    print(
        f"Validation F1  : "
        f"{best_score:.4f}"
    )

    print()
    print(
        "Evaluating the selected model "
        "on the untouched test set..."
    )

    test_predictions = (
        best_model.predict(
            x_test
        )
    )

    test_probabilities = (
        positive_class_probability(
            best_model,
            x_test,
        )
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities,
    )

    # Re-open the winning MLflow run and attach final test metrics.
    with mlflow.start_run(
        run_id=best_run_id
    ):
        mlflow.log_metrics(
            {
                (
                    f"test_{name}"
                ): value
                for (
                    name,
                    value,
                ) in (
                    test_metrics.items()
                )
            }
        )

        mlflow.set_tag(
            "selected_as_best",
            "true",
        )

    model_path = (
        models_dir
        / "best_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    metadata = {
        "project": (
            "Construction Safety Risk Predictor"
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "best_model": (
            best_model_name
        ),
        "selection_metric": (
            SELECTION_METRIC
        ),
        "validation_score": (
            best_score
        ),
        "mlflow_run_id": (
            best_run_id
        ),
        "model_path": (
            str(model_path)
        ),
        "features": (
            FEATURE_COLUMNS
        ),
        "target": (
            "behavior"
        ),
        "target_definition": {
            "0": "unsafe",
            "1": "safe",
        },
        "split_strategy": {
            "train_runs": (
                "0-79"
            ),
            "validation_runs": (
                "80-89"
            ),
            "test_runs": (
                "90-99"
            ),
        },
        "rows": {
            "total": (
                len(dataframe)
            ),
            "train": (
                len(train_df)
            ),
            "validation": (
                len(
                    validation_df
                )
            ),
            "test": (
                len(test_df)
            ),
        },
        "validation_metrics": (
            best_validation_metrics
        ),
        "test_metrics": (
            test_metrics
        ),
    }

    metadata_path = (
        models_dir
        / "model_metadata.json"
    )

    save_json(
        metadata,
        metadata_path,
    )

    print()
    print("=" * 72)
    print(
        "FINAL TEST RESULTS"
    )
    print("=" * 72)

    for (
        metric_name,
        metric_value,
    ) in test_metrics.items():
        print(
            f"{metric_name:<20}: "
            f"{metric_value:.4f}"
        )

    print("=" * 72)

    print(
        f"Best model saved to: "
        f"{model_path}"
    )

    print(
        f"Metadata saved to:   "
        f"{metadata_path}"
    )

    print(
        f"Comparison saved to: "
        f"{comparison_path}"
    )

    print(
        f"MLflow run ID:       "
        f"{best_run_id}"
    )

    return metadata


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train and compare safety "
            "behavior classifiers."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/safety_dataset.csv"
        ),
        help="Generated training dataset.",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(
            "models"
        ),
        help="Directory for model artifacts.",
    )

    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(
            "reports"
        ),
        help="Directory for training reports.",
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=(
            "sqlite:///mlflow.db"
        ),
        help="MLflow tracking URI.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=(
            "construction-safety-risk"
        ),
        help="MLflow experiment name.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_arguments()

    train_models(
        data_path=args.data,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=(
            args.experiment_name
        ),
    )


if __name__ == "__main__":
    main()
