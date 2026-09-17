"""
Unit tests for the agents module.
"""

import pytest
import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import Worker, Manager


class TestWorker:
    """Tests for the Worker agent."""

    def test_worker_initial_attributes(self):
        """Attributes are drawn from N(0.6, 0.1), within [0, 1]."""
        rng = random.Random(42)
        for _ in range(20):
            w = Worker(worker_id=0, rng=rng)
            for attr in (w.SA, w.SK, w.SN, w.BA, w.PBC):
                assert 0.0 <= attr <= 1.0

    def test_worker_decide_returns_dict(self):
        """decide() returns a dict with expected keys."""
        rng = random.Random(42)
        w = Worker(worker_id=0, rng=rng)
        result = w.decide()
        assert 'safe' in result
        assert 'behavior' in result
        assert 'SB' in result
        assert isinstance(result['safe'], bool)

    def test_worker_behavior_history(self):
        """Each decide() appends to behavior_history."""
        rng = random.Random(42)
        w = Worker(worker_id=0, rng=rng)
        for _ in range(5):
            w.decide()
        assert len(w.behavior_history) == 5


class TestManager:
    """Tests for the Manager agent."""

    def test_manager_initial_behaviors(self):
        """Manager behaviors are within [0.5, 1.0]."""
        rng = random.Random(42)
        m = Manager(manager_id=0, rng=rng)
        for attr in (m.ET, m.IR, m.SI, m.SM, m.HEM):
            assert 0.5 <= attr <= 1.0

    def test_manager_influence_raises_worker_attributes(self):
        """A manager with high behaviors improves worker's SK."""
        rng = random.Random(42)
        w = Worker(worker_id=0, rng=rng)
        w.SK = 0.3
        m = Manager(manager_id=0, rng=rng)
        m.ET, m.IR, m.SI, m.SM, m.HEM = 0.9, 0.9, 0.9, 0.9, 0.9
        m.apply_to_worker(w)
        assert w.SK > 0.3

    def test_manager_update_behaviors_safety_improving(self):
        """If safety improved, manager relaxes ET (Eq. 6)."""
        rng = random.Random(42)
        m = Manager(manager_id=0, rng=rng)
        m.ET = 0.8
        m.update_behaviors(prev_rate=0.5, current_rate=0.7)
        assert m.ET < 0.8

    def test_manager_update_behaviors_safety_dropping(self):
        """
        When safety drops, ET should decay MORE SLOWLY than when
        safety improves — because the correction term partially
        compensates the natural forgetting.
        """
        rng = random.Random(42)

        # Scenario A: safety improves
        m_improve = Manager(manager_id=0, rng=rng)
        m_improve.ET = 0.6
        m_improve.update_behaviors(prev_rate=0.5, current_rate=0.7)

        # Scenario B: safety drops
        m_drop = Manager(manager_id=0, rng=rng)
        m_drop.ET = 0.6
        m_drop.update_behaviors(prev_rate=0.7, current_rate=0.5)

        assert m_drop.ET > m_improve.ET


if __name__ == '__main__':
    pytest.main([__file__, '-v'])