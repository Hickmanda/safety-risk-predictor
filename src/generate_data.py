"""
Generate a supervised ML dataset from the construction safety ABM.

Default configuration:

100 simulation runs
x 100 workers
x 100 simulation days
= 1,000,000 observations

Each observation represents one worker decision during one day.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.model import SafetyModel


METADATA_COLUMNS = [
    "run_id",
    "seed",
    "worker_id",
]


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


TARGET_COLUMN = "behavior"


ALL_COLUMNS = (
    METADATA_COLUMNS
    + FEATURE_COLUMNS
    + [TARGET_COLUMN]
)


def validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate generated data before saving it.

    Early validation prevents corrupted simulation output from
    becoming training data later in the pipeline.
    """

    if dataframe.empty:
        raise ValueError(
            "Generated dataframe is empty."
        )

    missing_columns = (
        set(ALL_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if (
        dataframe[ALL_COLUMNS]
        .isnull()
        .any()
        .any()
    ):
        raise ValueError(
            "Generated dataset contains missing values."
        )

    normalized_columns = [
        "SA",
        "SK",
        "SN",
        "BA",
        "PBC",
        "reference_point",
        "intention",
    ]

    for column in normalized_columns:
        if not dataframe[
            column
        ].between(
            0.0,
            1.0,
        ).all():
            raise ValueError(
                f"{column} contains values "
                "outside [0, 1]."
            )

    if not dataframe[
        "alpha"
    ].between(
        0.1,
        2.0,
    ).all():
        raise ValueError(
            "alpha contains values outside its valid range."
        )

    if not dataframe[
        "beta"
    ].between(
        0.1,
        2.0,
    ).all():
        raise ValueError(
            "beta contains values outside its valid range."
        )

    if not dataframe[
        "lam"
    ].between(
        1.0,
        2.5,
    ).all():
        raise ValueError(
            "lam contains values outside its valid range."
        )

    behavior_values = set(
        dataframe[
            TARGET_COLUMN
        ].unique()
    )

    if not behavior_values.issubset(
        {0, 1}
    ):
        raise ValueError(
            "behavior must contain only 0 and 1."
        )


def generate_single_run(
    run_id: int,
    seed: int,
    num_workers: int,
    num_managers: int,
    num_days: int,
) -> pd.DataFrame:
    """
    Generate one independent ABM simulation.

    run_id and seed are stored as metadata so that later ML
    train/validation/test splits can be performed by simulation
    instead of randomly mixing correlated trajectories.
    """

    model = SafetyModel(
        num_workers=num_workers,
        num_managers=num_managers,
        seed=seed,
    )

    records: list[
        dict[str, Any]
    ] = []

    for _ in range(num_days):
        # Mesa 3.x requires step() to be called without custom arguments.
        model.step()

        day_records = (
            model.last_step_records
        )

        if not day_records:
            raise RuntimeError(
                "Model did not generate worker records."
            )

        for record in day_records:
            # Copy the dictionary so metadata does not mutate
            # model-owned records unexpectedly.
            enriched_record = (
                record.copy()
            )

            enriched_record[
                "run_id"
            ] = run_id

            enriched_record[
                "seed"
            ] = seed

            records.append(
                enriched_record
            )

    dataframe = pd.DataFrame(
        records,
        columns=ALL_COLUMNS,
    )

    expected_rows = (
        num_workers
        * num_days
    )

    if len(dataframe) != expected_rows:
        raise RuntimeError(
            f"Run {run_id} generated "
            f"{len(dataframe):,} rows; "
            f"expected {expected_rows:,}."
        )

    validate_dataframe(
        dataframe
    )

    return dataframe


def generate_dataset(
    output_path: Path,
    num_runs: int = 100,
    num_workers: int = 100,
    num_managers: int = 3,
    num_days: int = 100,
    start_seed: int = 0,
) -> dict[str, float | int]:
    """
    Generate the complete training dataset.

    Simulation runs are written to disk incrementally so the
    complete one-million-row dataset does not need to remain
    in memory.
    """

    if num_runs <= 0:
        raise ValueError(
            "num_runs must be greater than zero."
        )

    if num_workers <= 0:
        raise ValueError(
            "num_workers must be greater than zero."
        )

    if num_managers < 0:
        raise ValueError(
            "num_managers cannot be negative."
        )

    if num_days <= 0:
        raise ValueError(
            "num_days must be greater than zero."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    expected_rows = (
        num_runs
        * num_workers
        * num_days
    )

    total_rows = 0
    total_safe = 0

    print()
    print("=" * 72)
    print(
        "CONSTRUCTION SAFETY DATASET GENERATOR"
    )
    print("=" * 72)

    print(
        f"Simulation runs : {num_runs}"
    )

    print(
        f"Workers/run     : {num_workers}"
    )

    print(
        f"Managers/run    : {num_managers}"
    )

    print(
        f"Days/run        : {num_days}"
    )

    print(
        f"Expected rows   : "
        f"{expected_rows:,}"
    )

    print(
        f"Output          : "
        f"{output_path}"
    )

    print("=" * 72)

    try:
        for run_id in range(
            num_runs
        ):
            seed = (
                start_seed
                + run_id
            )

            dataframe = (
                generate_single_run(
                    run_id=run_id,
                    seed=seed,
                    num_workers=num_workers,
                    num_managers=num_managers,
                    num_days=num_days,
                )
            )

            run_rows = len(
                dataframe
            )

            safe_count = int(
                dataframe[
                    TARGET_COLUMN
                ].sum()
            )

            run_safe_rate = (
                safe_count
                / run_rows
            )

            dataframe.to_csv(
                temporary_path,
                mode=(
                    "w"
                    if run_id == 0
                    else "a"
                ),
                header=(
                    run_id == 0
                ),
                index=False,
                float_format="%.6f",
            )

            total_rows += run_rows
            total_safe += safe_count

            progress = (
                (run_id + 1)
                / num_runs
                * 100.0
            )

            print(
                f"[{run_id + 1:03d}/"
                f"{num_runs:03d}] "
                f"seed={seed:<4d} | "
                f"rows={total_rows:>10,} | "
                f"safe={run_safe_rate:>6.2%} | "
                f"progress={progress:>6.2f}%"
            )

        if total_rows != expected_rows:
            raise RuntimeError(
                f"Generated {total_rows:,} rows; "
                f"expected {expected_rows:,}."
            )

        if output_path.exists():
            output_path.unlink()

        temporary_path.replace(
            output_path
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise

    unsafe_count = (
        total_rows
        - total_safe
    )

    safe_rate = (
        total_safe
        / total_rows
    )

    unsafe_rate = (
        unsafe_count
        / total_rows
    )

    file_size_mb = (
        output_path.stat().st_size
        / (1024 * 1024)
    )

    print()
    print("=" * 72)
    print(
        "GENERATION COMPLETE"
    )
    print("=" * 72)

    print(
        f"Rows       : "
        f"{total_rows:,}"
    )

    print(
        f"Safe       : "
        f"{total_safe:,} "
        f"({safe_rate:.2%})"
    )

    print(
        f"Unsafe     : "
        f"{unsafe_count:,} "
        f"({unsafe_rate:.2%})"
    )

    print(
        f"File size  : "
        f"{file_size_mb:.2f} MB"
    )

    print(
        f"Saved      : "
        f"{output_path.resolve()}"
    )

    print("=" * 72)

    return {
        "rows": total_rows,
        "safe_rows": total_safe,
        "unsafe_rows": unsafe_count,
        "safe_rate": safe_rate,
        "unsafe_rate": unsafe_rate,
        "file_size_mb": file_size_mb,
    }


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate an ML dataset from "
            "the construction safety ABM."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of independent simulation runs.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Number of workers per simulation.",
    )

    parser.add_argument(
        "--managers",
        type=int,
        default=3,
        help="Number of managers per simulation.",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=100,
        help="Number of days per simulation.",
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Random seed used for the first simulation.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/safety_dataset.csv"
        ),
        help="Output CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_arguments()

    generate_dataset(
        output_path=args.output,
        num_runs=args.runs,
        num_workers=args.workers,
        num_managers=args.managers,
        num_days=args.days,
        start_seed=args.start_seed,
    )


if __name__ == "__main__":
    main()
