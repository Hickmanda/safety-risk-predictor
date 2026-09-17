"""
Agent definitions for the ABM simulation.
Based on Zhang et al. (2025, ASCE JCEM), Fig. 1 and Eq. 1–20.
"""

import random
import numpy as np

from src.cpt import compute_cpt_safe, compute_cpt_unsafe
from src import params


class Worker:
    """A construction worker agent with a 3-stage cognitive model."""

    def __init__(self, worker_id, rng=None):
        self.id = worker_id
        self.rng = rng or random.Random()

        self.SA = self._draw_attribute()
        self.SK = self._draw_attribute()
        self.SN = self._draw_attribute()
        self.BA = self._draw_attribute()
        self.PBC = self._draw_attribute()

        self.m = params.MEMORY_COEFFICIENT

        self.reference_point = params.REFERENCE_POINT_INIT
        self.alpha = params.ALPHA
        self.beta = params.BETA
        self.lam = params.LAMBDA

        self.prev_SN = self.SN
        self.prev_BA = self.BA
        self.prev_PBC = self.PBC

        self.behavior_history = []

    def _draw_attribute(self):
        val = self.rng.gauss(params.ATTRIBUTE_MEAN, params.ATTRIBUTE_STD)
        return max(0.0, min(1.0, val))

    def compute_intention(self):
        return (self.SN + self.BA + self.PBC) / 3.0

    def update_behavior_attributes(self):
        """Update dynamic CPT parameters (Eq. 16–19)."""
        eps = 1e-6

        if self.prev_SN > eps:
            sn_change = (self.SN - self.prev_SN) / self.prev_SN
            self.reference_point *= (1.0 + sn_change)

        if self.prev_BA > eps:
            ba_change = (self.BA - self.prev_BA) / self.prev_BA
            self.alpha *= (1.0 + ba_change)
            self.beta *= (1.0 + ba_change)
        else:
            ba_change = 0.0

        if self.prev_PBC > eps:
            pbc_change = (self.PBC - self.prev_PBC) / self.prev_PBC
            self.lam += (2.5 - self.lam) * pbc_change * ba_change

        self.reference_point = max(0.0, min(1.0, self.reference_point))
        self.alpha = max(0.1, min(2.0, self.alpha))
        self.beta = max(0.1, min(2.0, self.beta))
        self.lam = max(1.0, min(2.5, self.lam))

        self.prev_SN = self.SN
        self.prev_BA = self.BA
        self.prev_PBC = self.PBC

    def decide(self):
        """Decide safe vs unsafe behavior (Eq. 14–15, 20)."""
        cpt_s = compute_cpt_safe(
            reference_point=self.reference_point,
            alpha=self.alpha, beta=self.beta, lam=self.lam,
        )
        cpt_u = compute_cpt_unsafe(
            reference_point=self.reference_point,
            alpha=self.alpha, beta=self.beta, lam=self.lam,
        )

        diff = cpt_s - cpt_u
        sb = 1.0 / (1.0 + np.exp(-params.SIGMOID_K * diff))

        noisy_SA = self.SA + self.rng.gauss(0.0, 0.08)
        noisy_SK = self.SK + self.rng.gauss(0.0, 0.08)

        safe = (noisy_SA >= params.COGNITIVE_THRESHOLD and
                noisy_SK >= params.COGNITIVE_THRESHOLD and
                sb >= params.COGNITIVE_THRESHOLD)

        behavior = 1 if safe else 0
        self.behavior_history.append(behavior)

        return {
            "safe": safe,
            "behavior": behavior,
            "SA": self.SA,
            "SK": self.SK,
            "SB": sb,
            "intention": self.compute_intention(),
            "cpt_safe": cpt_s,
            "cpt_unsafe": cpt_u,
        }


class Manager:
    """A manager agent with 5 behavior levels: ET, IR, SI, SM, HEM."""

    def __init__(self, manager_id, rng=None):
        self.id = manager_id
        self.rng = rng or random.Random()

        self.ET = self._draw_behavior()
        self.IR = self._draw_behavior()
        self.SI = self._draw_behavior()
        self.SM = self._draw_behavior()
        self.HEM = self._draw_behavior()

        self.n = params.MEMORY_COEFFICIENT

    def _draw_behavior(self):
        val = self.rng.gauss(params.MANAGER_BEHAVIOR_MEAN,
                             params.MANAGER_BEHAVIOR_STD)
        return max(0.5, min(1.0, val))

    def apply_to_worker(self, worker):
        """Update worker's SA, SK, SN, BA, PBC — Eq. 1–5."""
        m = worker.m

        worker.SA = (1 - 1/m) * worker.SA + (1/m) * (
            params.OMEGA_SA_ET * self.ET +
            params.OMEGA_SA_IR * self.IR +
            params.OMEGA_SA_HEM * self.HEM
        )

        worker.SK = (1 - 1/m) * worker.SK + (1/m) * (
            params.OMEGA_SK_ET * self.ET +
            params.OMEGA_SK_SI * self.SI +
            params.OMEGA_SK_SM * self.SM +
            params.OMEGA_SK_HEM * self.HEM
        )

        worker.SN = (1 - 1/m) * worker.SN + (1/m) * (
            params.OMEGA_SN_IR * self.IR +
            params.OMEGA_SN_SM * self.SM +
            params.OMEGA_SN_HEM * self.HEM
        )

        worker.BA = (1 - 1/m) * worker.BA + (1/m) * (
            params.OMEGA_BA_IR * self.IR +
            params.OMEGA_BA_SM * self.SM
        )

        worker.PBC = (1 - 1/m) * worker.PBC + (1/m) * (
            params.OMEGA_PBC_ET * self.ET +
            params.OMEGA_PBC_HEM * self.HEM
        )

        for attr in ('SA', 'SK', 'SN', 'BA', 'PBC'):
            val = getattr(worker, attr)
            setattr(worker, attr, max(0.0, min(1.0, val)))

    def update_behaviors(self, prev_rate, current_rate):
        """Adjust ET and IR based on safety trend — Eq. 6–7."""
        if prev_rate > 1e-6:
            delta = (current_rate - prev_rate) / prev_rate
        else:
            delta = 0.0

        reaction = 0.15
        n = self.n

        self.ET = (1 - 1/n) * self.ET - (1/n) * self.ET * delta * reaction
        self.IR = (1 - 1/n) * self.IR - (1/n) * self.IR * delta * reaction

        self.ET = max(0.4, min(1.0, self.ET))
        self.IR = max(0.4, min(1.0, self.IR))


if __name__ == '__main__':
    print("=" * 60)
    print("Test: Worker + Manager interaction (Eq. 1–7)")
    print("=" * 60)

    rng = random.Random(42)
    worker = Worker(worker_id=0, rng=rng)
    manager = Manager(manager_id=0, rng=rng)

    for step in range(1, 21):
        manager.apply_to_worker(worker)
        worker.update_behavior_attributes()
        result = worker.decide()

        if step % 4 == 0 or step == 1:
            print(f"\nStep {step}:")
            print(f"  SA={worker.SA:.3f}  SK={worker.SK:.3f}  "
                  f"SN={worker.SN:.3f}  BA={worker.BA:.3f}  PBC={worker.PBC:.3f}")
            print(f"  -> {'SAFE' if result['safe'] else 'UNSAFE'}")