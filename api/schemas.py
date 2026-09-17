"""
Pydantic schemas for the prediction API.

The schemas provide strict validation at the API boundary so invalid
or out-of-distribution inputs cannot silently reach the ML model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class PredictionRequest(BaseModel):
    """
    Input features for one worker safety prediction.

    Intention is optional because it is mathematically derived from
    SN, BA and PBC in the original ABM. If omitted, the API calculates
    it automatically.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    SA: float = Field(
        ge=0.0,
        le=1.0,
        description="Situational Awareness",
    )

    SK: float = Field(
        ge=0.0,
        le=1.0,
        description="Safety Knowledge",
    )

    SN: float = Field(
        ge=0.0,
        le=1.0,
        description="Subjective Norm",
    )

    BA: float = Field(
        ge=0.0,
        le=1.0,
        description="Behavior Attitude",
    )

    PBC: float = Field(
        ge=0.0,
        le=1.0,
        description="Perceived Behavior Control",
    )

    reference_point: float = Field(
        ge=0.0,
        le=1.0,
    )

    alpha: float = Field(
        ge=0.1,
        le=2.0,
    )

    beta: float = Field(
        ge=0.1,
        le=2.0,
    )

    lam: float = Field(
        ge=1.0,
        le=2.5,
    )

    intention: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    day: int = Field(
        ge=0,
        le=99,
    )

    @model_validator(
        mode="after"
    )
    def calculate_or_validate_intention(
        self,
    ) -> "PredictionRequest":
        """
        Keep intention consistent with the ABM definition.

        Training data always uses:
            intention = mean(SN, BA, PBC)

        Accepting inconsistent values would create inputs that never
        existed in the training distribution.
        """

        expected_intention = (
            self.SN
            + self.BA
            + self.PBC
        ) / 3.0

        if self.intention is None:
            self.intention = (
                expected_intention
            )

            return self

        if abs(
            self.intention
            - expected_intention
        ) > 1e-4:
            raise ValueError(
                "intention must equal "
                "the mean of SN, BA and PBC."
            )

        return self


class PredictionResponse(BaseModel):
    """
    API response for one prediction.
    """

    behavior: int

    label: Literal[
        "safe",
        "unsafe",
    ]

    safe_probability: float

    unsafe_probability: float

    decision_threshold: float

    latency_ms: float


class HealthResponse(BaseModel):
    """
    Service health information.
    """

    status: Literal[
        "ok",
        "degraded",
    ]

    model_loaded: bool

    model_name: str | None = None

    detail: str | None = None


class ModelInfoResponse(BaseModel):
    """
    Public metadata about the deployed model.
    """

    model_name: str

    selection_metric: str

    validation_score: float

    decision_threshold: float

    feature_count: int

    features: list[str]

    target: str

    threshold_objective: str


class MetricsResponse(BaseModel):
    """
    Lightweight in-memory API monitoring metrics.

    These counters reset whenever the API process restarts.
    """

    total_predictions: int

    safe_predictions: int

    unsafe_predictions: int

    average_latency_ms: float

    uptime_seconds: float
