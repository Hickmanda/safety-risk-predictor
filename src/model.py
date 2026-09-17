"""
Mesa-based construction safety model.

This module extends the original Project 2 ABM with worker-level
record collection for machine-learning dataset generation.

The original simulation order is intentionally preserved.
"""

from __future__ import annotations

import random
from typing import Any

import mesa

from src.agents import Manager, Worker


class SafetyModel(mesa.Model):
    """
    Construction site safety simulation.

    Parameters
    ----------
    num_workers:
        Number of construction worker agents.
    num_managers:
        Number of manager agents.
    seed:
        Random seed used for reproducible simulations.
    """

    def __init__(
        self,
        num_workers: int = 100,
        num_managers: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)

        if num_workers <= 0:
            raise ValueError(
                "num_workers must be greater than zero."
            )

        if num_managers < 0:
            raise ValueError(
                "num_managers cannot be negative."
            )

        self.rng = random.Random(seed)

        self.num_workers = num_workers
        self.num_managers = num_managers

        self.workers = [
            Worker(
                worker_id=i,
                rng=self.rng,
            )
            for i in range(num_workers)
        ]

        self.managers = [
            Manager(
                manager_id=i,
                rng=self.rng,
            )
            for i in range(num_managers)
        ]

        self.safety_rate_history: list[float] = []
        self.avg_SA_history: list[float] = []
        self.avg_SK_history: list[float] = []
        self.avg_reference_point_history: list[float] = []

        self.smoothed_rate: float | None = None
        self.prev_smoothed_rate: float | None = None

        self.step_count = 0

        # Mesa 3.x wraps Model.step(), so step() should not receive
        # custom arguments. Instead, records from the most recent
        # simulation day are stored here.
        self.last_step_records: list[
            dict[str, Any]
        ] = []

    def step(self) -> None:
        """
        Execute one simulation day.

        Worker-level ML records are stored in last_step_records.

        Features are captured after each worker updates its CPT
        parameters and makes the current decision, but before managers
        modify worker attributes for the following day.

        This ordering prevents feature/label misalignment.
        """

        decisions: list[int] = []
        records: list[dict[str, Any]] = []

        # Workers update CPT parameters and make today's decisions.
        for worker in self.workers:
            worker.update_behavior_attributes()

            result = worker.decide()

            behavior = int(
                result["behavior"]
            )

            decisions.append(
                behavior
            )

            records.append(
                {
                    "worker_id": worker.id,
                    "SA": float(
                        worker.SA
                    ),
                    "SK": float(
                        worker.SK
                    ),
                    "SN": float(
                        worker.SN
                    ),
                    "BA": float(
                        worker.BA
                    ),
                    "PBC": float(
                        worker.PBC
                    ),
                    "reference_point": float(
                        worker.reference_point
                    ),
                    "alpha": float(
                        worker.alpha
                    ),
                    "beta": float(
                        worker.beta
                    ),
                    "lam": float(
                        worker.lam
                    ),
                    "intention": float(
                        worker.compute_intention()
                    ),
                    "day": self.step_count,
                    "behavior": behavior,
                }
            )

        # Store the records before manager actions change worker states.
        self.last_step_records = records

        # Calculate today's site-level safety rate.
        safety_rate = (
            sum(decisions)
            / len(decisions)
        )

        # Smooth the safety rate exactly as in Project 2.
        if self.smoothed_rate is None:
            self.smoothed_rate = (
                safety_rate
            )

            self.prev_smoothed_rate = (
                safety_rate
            )

        else:
            self.prev_smoothed_rate = (
                self.smoothed_rate
            )

            self.smoothed_rate = (
                0.2 * safety_rate
                + 0.8 * self.smoothed_rate
            )

        # Managers react to the current site safety trend.
        for manager in self.managers:
            manager.update_behaviors(
                self.prev_smoothed_rate,
                self.smoothed_rate,
            )

        # Aggregate manager behavior into one team influence.
        if self.managers:
            team_ET = sum(
                manager.ET
                for manager in self.managers
            ) / len(self.managers)

            team_IR = sum(
                manager.IR
                for manager in self.managers
            ) / len(self.managers)

            team_SI = sum(
                manager.SI
                for manager in self.managers
            ) / len(self.managers)

            team_SM = sum(
                manager.SM
                for manager in self.managers
            ) / len(self.managers)

            team_HEM = sum(
                manager.HEM
                for manager in self.managers
            ) / len(self.managers)

            # Preserve this temporary Manager construction from
            # Project 2 so random-number consumption stays identical.
            team = Manager(
                manager_id=-1,
                rng=self.rng,
            )

            (
                team.ET,
                team.IR,
                team.SI,
                team.SM,
                team.HEM,
            ) = (
                team_ET,
                team_IR,
                team_SI,
                team_SM,
                team_HEM,
            )

            # Manager influence changes worker states for the next day.
            for worker in self.workers:
                team.apply_to_worker(
                    worker
                )

        # Preserve the statistics collected by Project 2.
        self.safety_rate_history.append(
            safety_rate
        )

        self.avg_SA_history.append(
            sum(
                worker.SA
                for worker in self.workers
            )
            / len(self.workers)
        )

        self.avg_SK_history.append(
            sum(
                worker.SK
                for worker in self.workers
            )
            / len(self.workers)
        )

        self.avg_reference_point_history.append(
            sum(
                worker.reference_point
                for worker in self.workers
            )
            / len(self.workers)
        )

        self.step_count += 1

    def run(
        self,
        steps: int = 100,
    ) -> None:
        """
        Run the simulation for a fixed number of days.
        """

        if steps <= 0:
            raise ValueError(
                "steps must be greater than zero."
            )

        for _ in range(steps):
            self.step()


def main() -> None:
    """
    Run the original Project 2 configuration as a regression check.
    """

    print("=" * 60)
    print(
        "Construction Safety ABM"
    )
    print("=" * 60)

    model = SafetyModel(
        num_workers=100,
        num_managers=3,
        seed=42,
    )

    model.run(
        steps=100
    )

    print(
        f"Initial safety rate: "
        f"{model.safety_rate_history[0]:.2%}"
    )

    print(
        f"Final safety rate:   "
        f"{model.safety_rate_history[-1]:.2%}"
    )


if __name__ == "__main__":
    main()
