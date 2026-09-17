"""
Tests for the construction safety ABM.
"""

import pytest

from src.model import SafetyModel


def test_worker_count() -> None:
    """
    The model should create the requested number of workers.
    """

    model = SafetyModel(
        num_workers=10,
        num_managers=3,
        seed=42,
    )

    assert len(
        model.workers
    ) == 10


def test_manager_count() -> None:
    """
    The model should create the requested number of managers.
    """

    model = SafetyModel(
        num_workers=10,
        num_managers=3,
        seed=42,
    )

    assert len(
        model.managers
    ) == 3


def test_step_counter() -> None:
    """
    One simulation step should increase the day counter by one.
    """

    model = SafetyModel(
        num_workers=10,
        num_managers=3,
        seed=42,
    )

    assert model.step_count == 0

    model.step()

    assert model.step_count == 1


def test_record_collection() -> None:
    """
    Each simulation step should store one record per worker.
    """

    model = SafetyModel(
        num_workers=10,
        num_managers=3,
        seed=42,
    )

    model.step()

    records = (
        model.last_step_records
    )

    assert len(
        records
    ) == 10


def test_record_contains_target() -> None:
    """
    Every record should contain a binary behavior target.
    """

    model = SafetyModel(
        num_workers=10,
        num_managers=3,
        seed=42,
    )

    model.step()

    records = (
        model.last_step_records
    )

    for record in records:
        assert (
            "behavior"
            in record
        )

        assert record[
            "behavior"
        ] in {
            0,
            1,
        }


def test_record_contains_features() -> None:
    """
    Every record should contain all required ML features.
    """

    model = SafetyModel(
        num_workers=5,
        num_managers=3,
        seed=42,
    )

    model.step()

    required_features = {
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
    }

    for record in (
        model.last_step_records
    ):
        assert required_features.issubset(
            record.keys()
        )


def test_record_day_matches_step() -> None:
    """
    Records from the first step should belong to day zero.
    """

    model = SafetyModel(
        num_workers=5,
        num_managers=3,
        seed=42,
    )

    model.step()

    for record in (
        model.last_step_records
    ):
        assert record[
            "day"
        ] == 0

    model.step()

    for record in (
        model.last_step_records
    ):
        assert record[
            "day"
        ] == 1


def test_invalid_worker_count() -> None:
    """
    A model cannot contain zero workers.
    """

    with pytest.raises(
        ValueError
    ):
        SafetyModel(
            num_workers=0,
            num_managers=3,
        )


def test_invalid_manager_count() -> None:
    """
    A negative manager count should be rejected.
    """

    with pytest.raises(
        ValueError
    ):
        SafetyModel(
            num_workers=10,
            num_managers=-1,
        )
