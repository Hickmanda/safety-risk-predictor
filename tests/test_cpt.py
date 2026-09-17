"""
Unit tests for the CPT module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cpt import (
    value_function,
    weight_gain,
    weight_loss,
    compute_cpt_safe,
    compute_cpt_unsafe,
    prospect,
)


class TestValueFunction:
    """Tests for Eq. 10 — the CPT value function."""

    def test_zero_deviation(self):
        """At the reference point, the value should be 0."""
        result = value_function(0.6, 0.6)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_gain_is_positive(self):
        """A gain above the reference point is positive."""
        result = value_function(0.8, 0.6)
        assert result > 0

    def test_loss_is_negative(self):
        """A loss below the reference point is negative."""
        result = value_function(0.4, 0.6)
        assert result < 0

    def test_loss_aversion(self):
        """Losses hurt more than equivalent gains."""
        gain = value_function(0.7, 0.6)
        loss = value_function(0.5, 0.6)
        assert abs(loss) > gain


class TestWeightFunctions:
    """Tests for Eq. 12–13 — probability weighting."""

    def test_gain_weight_monotonic(self):
        p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        w = weight_gain(p)
        assert np.all(np.diff(w) > 0)

    def test_loss_weight_monotonic(self):
        p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        w = weight_loss(p)
        assert np.all(np.diff(w) > 0)

    def test_gain_weight_bounds(self):
        p = np.linspace(0.01, 0.99, 20)
        w = weight_gain(p)
        assert np.all(w >= 0) and np.all(w <= 1)


class TestCPTSafeUnsafe:
    """Tests for Eq. 14–15."""

    def test_cpt_safe_is_negative_at_reference(self):
        result = compute_cpt_safe(reference_point=0.6)
        assert result < 0

    def test_cpt_unsafe_is_very_negative(self):
        result = compute_cpt_unsafe(reference_point=0.6)
        assert result < compute_cpt_safe(reference_point=0.6)

    def test_cpt_monotonic_in_reference(self):
        cpt_low = compute_cpt_safe(reference_point=0.4)
        cpt_high = compute_cpt_safe(reference_point=0.8)
        assert cpt_high < cpt_low


class TestProspect:
    """Tests for the generic CPT function (Eq. 9)."""

    def test_pure_gain_prospect(self):
        result = prospect(
            outcomes=[0.5, 0.8],
            probabilities=[0.5, 0.5],
            reference_point=0.0,
        )
        assert result > 0

    def test_pure_loss_prospect(self):
        result = prospect(
            outcomes=[-0.5, -0.8],
            probabilities=[0.5, 0.5],
            reference_point=0.0,
        )
        assert result < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])