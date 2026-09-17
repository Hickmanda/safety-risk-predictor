"""
Tests for simulation dataset generation.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.generate_data import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    generate_dataset,
    generate_single_run,
)


def test_feature_count() -> None:
    """
    The final ML model should use exactly 11 features.
    """

    assert len(FEATURE_COLUMNS) == 11


def test_single_run_row_count() -> None:
    """
    Row count should equal workers multiplied by days.
    """

    dataframe = generate_single_run(
        run_id=0,
        seed=42,
        num_workers=5,
        num_managers=3,
        num_days=4,
    )

    assert len(dataframe) == 20


def test_dataset_schema() -> None:
    """
    Generated data should contain exactly the expected columns.
    """

    dataframe = generate_single_run(
        run_id=0,
        seed=42,
        num_workers=5,
        num_managers=3,
        num_days=3,
    )

    assert list(
        dataframe.columns
    ) == ALL_COLUMNS


def test_behavior_is_binary() -> None:
    """
    The target variable should contain only zero and one.
    """

    dataframe = generate_single_run(
        run_id=0,
        seed=42,
        num_workers=20,
        num_managers=3,
        num_days=10,
    )

    values = set(
        dataframe[
            "behavior"
        ].unique()
    )

    assert values.issubset(
        {0, 1}
    )


def test_worker_ids() -> None:
    """
    Worker identifiers should remain stable within each run.
    """

    dataframe = generate_single_run(
        run_id=0,
        seed=42,
        num_workers=5,
        num_managers=3,
        num_days=4,
    )

    assert set(
        dataframe[
            "worker_id"
        ].unique()
    ) == {
        0,
        1,
        2,
        3,
        4,
    }


def test_days() -> None:
    """
    Simulation days should start at zero.
    """

    dataframe = generate_single_run(
        run_id=0,
        seed=42,
        num_workers=5,
        num_managers=3,
        num_days=5,
    )

    assert set(
        dataframe[
            "day"
        ].unique()
    ) == {
        0,
        1,
        2,
        3,
        4,
    }


def test_normalized_features() -> None:
    """
    Normalized cognitive features should remain inside [0, 1].
    """

    dataframe = generate_single_run(
        run_id=0,
        seed=42,
        num_workers=10,
        num_managers=3,
        num_days=10,
    )

    columns = [
        "SA",
        "SK",
        "SN",
        "BA",
        "PBC",
        "reference_point",
        "intention",
    ]

    for column in columns:
        assert dataframe[
            column
        ].between(
            0.0,
            1.0,
        ).all()


def test_cpt_parameter_ranges() -> None:
    """
    Dynamic CPT parameters should respect model boundaries.
    """

    dataframe = generate_single_run(
        run_id=0,
        seed=42,
        num_workers=10,
        num_managers=3,
        num_days=20,
    )

    assert dataframe[
        "alpha"
    ].between(
        0.1,
        2.0,
    ).all()

    assert dataframe[
        "beta"
    ].between(
        0.1,
        2.0,
    ).all()

    assert dataframe[
        "lam"
    ].between(
        1.0,
        2.5,
    ).all()


def test_intention_formula() -> None:
    """
    Intention should equal the mean of SN, BA and PBC.
    """

    dataframe = generate_single_run(
        run_id=0,
        seed=42,
        num_workers=10,
        num_managers=3,
        num_days=5,
    )

    expected = (
        dataframe["SN"]
        + dataframe["BA"]
        + dataframe["PBC"]
    ) / 3.0

    difference = (
        dataframe["intention"]
        - expected
    ).abs()

    assert (
        difference < 1e-10
    ).all()


def test_same_seed_is_reproducible() -> None:
    """
    Identical seeds should produce identical trajectories.
    """

    first = generate_single_run(
        run_id=0,
        seed=123,
        num_workers=10,
        num_managers=3,
        num_days=5,
    )

    second = generate_single_run(
        run_id=0,
        seed=123,
        num_workers=10,
        num_managers=3,
        num_days=5,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_different_seeds_are_different() -> None:
    """
    Different seeds should produce different trajectories.
    """

    first = generate_single_run(
        run_id=0,
        seed=1,
        num_workers=10,
        num_managers=3,
        num_days=5,
    )

    second = generate_single_run(
        run_id=1,
        seed=2,
        num_workers=10,
        num_managers=3,
        num_days=5,
    )

    assert not first[
        FEATURE_COLUMNS
    ].equals(
        second[
            FEATURE_COLUMNS
        ]
    )


def test_generate_csv(
    tmp_path: Path,
) -> None:
    """
    Dataset generation should create a readable CSV file.
    """

    output_path = (
        tmp_path
        / "dataset.csv"
    )

    result = generate_dataset(
        output_path=output_path,
        num_runs=2,
        num_workers=5,
        num_managers=3,
        num_days=4,
        start_seed=0,
    )

    assert output_path.exists()

    dataframe = pd.read_csv(
        output_path
    )

    assert len(dataframe) == 40
    assert result["rows"] == 40


def test_invalid_run_count(
    tmp_path: Path,
) -> None:
    """
    Zero simulation runs should be rejected.
    """

    with pytest.raises(ValueError):
        generate_dataset(
            output_path=(
                tmp_path
                / "dataset.csv"
            ),
            num_runs=0,
        )
