"""
Tests for the FastAPI prediction service.

Tests use a deterministic fake classifier so CI does not require
the locally trained model artifact.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import (
    TestClient,
)

import api.main as api_main


class FakeModel:
    """
    Minimal classifier implementing predict_proba().
    """

    def predict_proba(
        self,
        features,
    ) -> np.ndarray:
        """
        Return a deterministic 75% SAFE probability.
        """

        rows = len(
            features
        )

        return np.tile(
            np.array(
                [
                    [
                        0.25,
                        0.75,
                    ],
                ]
            ),
            (
                rows,
                1,
            ),
        )


@pytest.fixture(
    autouse=True
)
def fake_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Inject deterministic model artifacts for every API test.
    """

    metadata = {
        "best_model": "xgboost",
        "selection_metric": (
            "f1_macro"
        ),
        "validation_score": (
            0.6126
        ),
        "features": (
            api_main.FEATURE_COLUMNS
        ),
        "target": "behavior",
        "decision_threshold": (
            0.64
        ),
        "threshold_selection": {
            "objective": (
                "macro F1"
            ),
        },
    }

    monkeypatch.setattr(
        api_main,
        "_model",
        FakeModel(),
    )

    monkeypatch.setattr(
        api_main,
        "_metadata",
        metadata,
    )

    monkeypatch.setattr(
        api_main,
        "_total_predictions",
        0,
    )

    monkeypatch.setattr(
        api_main,
        "_safe_predictions",
        0,
    )

    monkeypatch.setattr(
        api_main,
        "_unsafe_predictions",
        0,
    )

    monkeypatch.setattr(
        api_main,
        "_total_latency_ms",
        0.0,
    )


@pytest.fixture
def client() -> TestClient:
    """
    Create the FastAPI test client.
    """

    return TestClient(
        api_main.app
    )


@pytest.fixture
def valid_payload() -> dict[
    str,
    float | int,
]:
    """
    Return one valid prediction request.

    Intention is intentionally omitted so the API derives it.
    """

    return {
        "SA": 0.70,
        "SK": 0.75,
        "SN": 0.68,
        "BA": 0.72,
        "PBC": 0.70,
        "reference_point": 0.60,
        "alpha": 0.88,
        "beta": 0.88,
        "lam": 1.18,
        "day": 50,
    }


def test_root(
    client: TestClient,
) -> None:
    """
    Root endpoint should expose basic service information.
    """

    response = client.get(
        "/"
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "status"
        ]
        == "running"
    )


def test_health(
    client: TestClient,
) -> None:
    """
    Health endpoint should report the injected model as loaded.
    """

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert body[
        "status"
    ] == "ok"

    assert body[
        "model_loaded"
    ] is True


def test_prediction(
    client: TestClient,
    valid_payload: dict[
        str,
        float | int,
    ],
) -> None:
    """
    A 75% SAFE probability should exceed the 0.64 threshold.
    """

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert body[
        "behavior"
    ] == 1

    assert body[
        "label"
    ] == "safe"

    assert body[
        "safe_probability"
    ] == pytest.approx(
        0.75
    )

    assert body[
        "decision_threshold"
    ] == pytest.approx(
        0.64
    )


def test_intention_is_optional(
    client: TestClient,
    valid_payload: dict[
        str,
        float | int,
    ],
) -> None:
    """
    The API should derive intention automatically.
    """

    assert (
        "intention"
        not in valid_payload
    )

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert (
        response.status_code
        == 200
    )


def test_inconsistent_intention_rejected(
    client: TestClient,
    valid_payload: dict[
        str,
        float | int,
    ],
) -> None:
    """
    Inputs inconsistent with the ABM feature definition are invalid.
    """

    valid_payload[
        "intention"
    ] = 0.10

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert (
        response.status_code
        == 422
    )


def test_feature_range_validation(
    client: TestClient,
    valid_payload: dict[
        str,
        float | int,
    ],
) -> None:
    """
    Normalized features outside [0, 1] should be rejected.
    """

    valid_payload[
        "SA"
    ] = 1.5

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert (
        response.status_code
        == 422
    )


def test_unknown_fields_rejected(
    client: TestClient,
    valid_payload: dict[
        str,
        float | int,
    ],
) -> None:
    """
    Unexpected request fields should not be silently ignored.
    """

    valid_payload[
        "secret_feature"
    ] = 123

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert (
        response.status_code
        == 422
    )


def test_model_info(
    client: TestClient,
) -> None:
    """
    Model metadata should be exposed through the API.
    """

    response = client.get(
        "/model-info"
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert body[
        "model_name"
    ] == "xgboost"

    assert body[
        "decision_threshold"
    ] == pytest.approx(
        0.64
    )

    assert body[
        "feature_count"
    ] == 11


def test_metrics_increment(
    client: TestClient,
    valid_payload: dict[
        str,
        float | int,
    ],
) -> None:
    """
    Prediction calls should increment monitoring counters.
    """

    client.post(
        "/predict",
        json=valid_payload,
    )

    client.post(
        "/predict",
        json=valid_payload,
    )

    response = client.get(
        "/metrics"
    )

    body = response.json()

    assert body[
        "total_predictions"
    ] == 2

    assert body[
        "safe_predictions"
    ] == 2

    assert body[
        "unsafe_predictions"
    ] == 0

    assert body[
        "average_latency_ms"
    ] >= 0.0
