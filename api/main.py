"""
Production FastAPI service for construction safety prediction.

The API loads the trained XGBoost model and model metadata lazily.
Lazy loading keeps imports and automated tests independent from the
presence of local model artifacts.

Endpoints
---------
GET  /
GET  /health
GET  /model-info
GET  /metrics
POST /predict
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
    Response,
)

from api.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
# Production inference features in the exact order used during training.
# Keeping this list lightweight prevents the API from importing the
# Mesa-based simulation stack at runtime.
FEATURE_COLUMNS = [
    "SA",
    "SK",
    "SN",
    "BA",
    "PBC",
    "reference_point",
    "alpha",
    "beta",
    "lam",
    "intention",
    "day",
]

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "models/best_model.pkl",
    )
)

METADATA_PATH = Path(
    os.getenv(
        "MODEL_METADATA_PATH",
        "models/model_metadata.json",
    )
)

app = FastAPI(
    title="Construction Safety Risk Predictor",
    description=(
        "Production REST API for predicting safe or unsafe "
        "construction worker behavior from ABM-derived features."
    ),
    version="0.1.0",
)


_model: Any | None = None
_metadata: dict[str, Any] | None = None

_model_lock = threading.Lock()
_metrics_lock = threading.Lock()

_started_at = time.time()

_total_predictions = 0
_safe_predictions = 0
_unsafe_predictions = 0
_total_latency_ms = 0.0


def load_artifacts() -> tuple[
    Any,
    dict[str, Any],
]:
    """
    Load the trained model and metadata once.

    The lock prevents two simultaneous first requests from loading
    duplicate copies of the model into memory.
    """

    global _model
    global _metadata

    if (
        _model is not None
        and _metadata is not None
    ):
        return (
            _model,
            _metadata,
        )

    with _model_lock:
        if (
            _model is not None
            and _metadata is not None
        ):
            return (
                _model,
                _metadata,
            )

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found: "
                f"{MODEL_PATH}"
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Model metadata not found: "
                f"{METADATA_PATH}"
            )

        model = joblib.load(
            MODEL_PATH
        )

        with METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )

        required_metadata = {
            "best_model",
            "selection_metric",
            "validation_score",
            "features",
            "target",
            "decision_threshold",
        }

        missing = (
            required_metadata
            - set(
                metadata.keys()
            )
        )

        if missing:
            raise ValueError(
                "Model metadata is missing: "
                f"{sorted(missing)}"
            )

        if metadata[
            "features"
        ] != FEATURE_COLUMNS:
            raise ValueError(
                "Model feature schema does not "
                "match the API feature schema."
            )

        _model = model
        _metadata = metadata

    return (
        _model,
        _metadata,
    )


def get_artifacts_or_503() -> tuple[
    Any,
    dict[str, Any],
]:
    """
    Convert artifact-loading failures into a service error.
    """

    try:
        return load_artifacts()

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction model is unavailable: "
                f"{error}"
            ),
        ) from error


def create_feature_frame(
    request: PredictionRequest,
) -> pd.DataFrame:
    """
    Build one model input row in the exact training feature order.
    """

    values = request.model_dump()

    row = {
        feature: values[
            feature
        ]
        for feature in FEATURE_COLUMNS
    }

    return pd.DataFrame(
        [
            row,
        ],
        columns=FEATURE_COLUMNS,
    )


def update_metrics(
    behavior: int,
    latency_ms: float,
) -> None:
    """
    Update lightweight prediction counters safely.
    """

    global _total_predictions
    global _safe_predictions
    global _unsafe_predictions
    global _total_latency_ms

    with _metrics_lock:
        _total_predictions += 1

        _total_latency_ms += (
            latency_ms
        )

        if behavior == 1:
            _safe_predictions += 1
        else:
            _unsafe_predictions += 1


@app.get("/")
def root() -> dict[str, str]:
    """
    Basic service landing endpoint.
    """

    return {
        "service": (
            "Construction Safety Risk Predictor"
        ),
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(
    response: Response,
) -> HealthResponse:
    """
    Verify that the API process and model artifact are available.
    """

    try:
        _, metadata = (
            load_artifacts()
        )

        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=metadata[
                "best_model"
            ],
        )

    except Exception as error:
        response.status_code = 503

        return HealthResponse(
            status="degraded",
            model_loaded=False,
            detail=str(error),
        )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    """
    Return metadata for the deployed model.
    """

    _, metadata = (
        get_artifacts_or_503()
    )

    threshold_data = metadata.get(
        "threshold_selection",
        {},
    )

    return ModelInfoResponse(
        model_name=metadata[
            "best_model"
        ],
        selection_metric=metadata[
            "selection_metric"
        ],
        validation_score=float(
            metadata[
                "validation_score"
            ]
        ),
        decision_threshold=float(
            metadata[
                "decision_threshold"
            ]
        ),
        feature_count=len(
            FEATURE_COLUMNS
        ),
        features=FEATURE_COLUMNS,
        target=metadata[
            "target"
        ],
        threshold_objective=(
            threshold_data.get(
                "objective",
                "macro F1",
            )
        ),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Predict whether worker behavior is safe or unsafe.

    The classifier outputs P(safe). The production threshold was
    optimized on validation simulations and is stored in metadata.
    """

    model, metadata = (
        get_artifacts_or_503()
    )

    start = time.perf_counter()

    features = (
        create_feature_frame(
            request
        )
    )

    probabilities = (
        model.predict_proba(
            features
        )
    )

    if probabilities.shape != (
        1,
        2,
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "Model returned an unexpected "
                "probability shape."
            ),
        )

    safe_probability = float(
        probabilities[
            0,
            1,
        ]
    )

    safe_probability = max(
        0.0,
        min(
            1.0,
            safe_probability,
        ),
    )

    unsafe_probability = (
        1.0
        - safe_probability
    )

    threshold = float(
        metadata[
            "decision_threshold"
        ]
    )

    behavior = int(
        safe_probability
        >= threshold
    )

    label = (
        "safe"
        if behavior == 1
        else "unsafe"
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    update_metrics(
        behavior=behavior,
        latency_ms=latency_ms,
    )

    return PredictionResponse(
        behavior=behavior,
        label=label,
        safe_probability=(
            safe_probability
        ),
        unsafe_probability=(
            unsafe_probability
        ),
        decision_threshold=(
            threshold
        ),
        latency_ms=latency_ms,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
)
def metrics() -> MetricsResponse:
    """
    Return lightweight runtime monitoring metrics.
    """

    with _metrics_lock:
        total = (
            _total_predictions
        )

        average_latency = (
            _total_latency_ms
            / total
            if total > 0
            else 0.0
        )

        safe = (
            _safe_predictions
        )

        unsafe = (
            _unsafe_predictions
        )

    uptime = (
        time.time()
        - _started_at
    )

    return MetricsResponse(
        total_predictions=total,
        safe_predictions=safe,
        unsafe_predictions=unsafe,
        average_latency_ms=(
            average_latency
        ),
        uptime_seconds=uptime,
    )
