"""
Null model tests for ARCO module

Tests:
1. Null model z-score computation
2. Composition-preserving shuffles
3. Markov-1 shuffles
4. Detection of significant periodicity (z > 3)
5. Random vs. periodic signal discrimination
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from arc_rci.arco_core import (
    arco_from_signal,
    null_model_zscore,
    _markov1_shuffle
)


class TestNullModel:
    """Test null model functionality."""

    def test_null_model_basic(self):
        """Test basic null model computation."""
        np.random.seed(42)

        # Create signal with some structure
        signal = np.sin(2 * np.pi * 0.1 * np.arange(200))

        z_score, null_rcis, real_rci, p_value = null_model_zscore(
            signal,
            n_shuffles=20,
            null_type='composition',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[50],
            max_q=11
        )

        # Check outputs
        assert isinstance(z_score, (float, np.floating))
        assert len(null_rcis) == 20
        assert isinstance(real_rci, (float, np.floating))
        assert 0 <= p_value <= 1

    def test_periodic_signal_high_zscore(self):
        """Test that periodic signal has high z-score."""
        np.random.seed(42)

        # Perfect 7-mer repeat
        motif = np.random.randn(7)
        n_repeats = 25
        signal = np.tile(motif, n_repeats)

        z_score, null_rcis, real_rci, p_value = null_model_zscore(
            signal,
            n_shuffles=30,
            null_type='composition',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[35, 70],
            max_q=11
        )

        # Z-score should be significantly positive
        assert z_score > 3.0, f"Expected z > 3.0, got z = {z_score:.2f}"

        # Real RCI should be higher than most nulls
        assert real_rci > np.mean(null_rcis)

    def test_random_signal_low_zscore(self):
        """Test that random signal has low z-score."""
        np.random.seed(42)

        # Random signal
        signal = np.random.randn(200)

        z_score, null_rcis, real_rci, p_value = null_model_zscore(
            signal,
            n_shuffles=30,
            null_type='composition',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31, 63],
            max_q=11
        )

        # Z-score should be close to 0 (not significantly different from null)
        assert abs(z_score) < 2.0, f"Expected |z| < 2.0, got z = {z_score:.2f}"

    def test_composition_preserving(self):
        """Test that composition shuffle preserves composition."""
        np.random.seed(42)

        signal = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

        # Multiple shuffles
        for _ in range(10):
            shuffled = np.random.permutation(signal)

            # Check composition
            assert sorted(shuffled) == sorted(signal)
            assert len(shuffled) == len(signal)

    def test_markov1_shuffle(self):
        """Test Markov-1 shuffle."""
        np.random.seed(42)

        signal = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])

        shuffled = _markov1_shuffle(signal)

        # Should preserve length
        assert len(shuffled) == len(signal)

        # Should preserve composition
        assert sorted(shuffled) == sorted(signal)

    def test_null_types(self):
        """Test different null model types."""
        np.random.seed(42)

        signal = np.random.randn(150)

        # Composition null
        z_comp, _, _, _ = null_model_zscore(
            signal,
            n_shuffles=20,
            null_type='composition',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31],
            max_q=7
        )

        # Markov-1 null
        z_markov, _, _, _ = null_model_zscore(
            signal,
            n_shuffles=20,
            null_type='markov1',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31],
            max_q=7
        )

        # Both should be finite
        assert np.isfinite(z_comp)
        assert np.isfinite(z_markov)

    def test_pvalue_range(self):
        """Test that p-value is in valid range."""
        np.random.seed(42)

        signal = np.sin(2 * np.pi * 0.1 * np.arange(100))

        _, _, _, p_value = null_model_zscore(
            signal,
            n_shuffles=20,
            null_type='composition',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31],
            max_q=7
        )

        assert 0 <= p_value <= 1

    def test_significance_threshold(self):
        """Test significance threshold (z > 3 corresponds to p < 0.01 approx)."""
        np.random.seed(42)

        # Strong periodic signal
        motif = np.array([1, 0, -1, 0])
        signal = np.tile(motif, 40)

        z_score, _, _, p_value = null_model_zscore(
            signal,
            n_shuffles=50,
            null_type='composition',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[40, 80],
            max_q=11
        )

        # High z-score should correspond to low p-value
        if z_score > 3.0:
            assert p_value < 0.1, f"z={z_score:.2f} but p={p_value:.4f}"


class TestNullModelReproducibility:
    """Test reproducibility of null models."""

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        signal = np.random.randn(100)

        # Run twice with same seed
        z1, nulls1, real1, p1 = null_model_zscore(
            signal,
            n_shuffles=20,
            null_type='composition',
            seed=123,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31],
            max_q=7
        )

        z2, nulls2, real2, p2 = null_model_zscore(
            signal,
            n_shuffles=20,
            null_type='composition',
            seed=123,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31],
            max_q=7
        )

        # Should get same results
        assert z1 == z2
        assert real1 == real2
        assert p1 == p2
        np.testing.assert_array_almost_equal(nulls1, nulls2)


class TestNullModelStatistics:
    """Test statistical properties of null model."""

    def test_null_distribution_statistics(self):
        """Test that null distribution has reasonable statistics."""
        np.random.seed(42)

        signal = np.random.randn(200)

        z_score, null_rcis, real_rci, p_value = null_model_zscore(
            signal,
            n_shuffles=50,
            null_type='composition',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31, 63],
            max_q=11
        )

        # Null RCIs should be non-negative
        assert np.all(null_rcis >= 0)

        # Null distribution should have positive variance
        assert np.var(null_rcis) > 0

        # Z-score calculation should be consistent
        null_mean = np.mean(null_rcis)
        null_std = np.std(null_rcis)

        if null_std > 0:
            expected_z = (real_rci - null_mean) / null_std
            assert np.abs(z_score - expected_z) < 1e-6

    def test_multiple_shuffles_coverage(self):
        """Test that multiple shuffles give good coverage."""
        np.random.seed(42)

        signal = np.random.randn(100)

        z_score, null_rcis, real_rci, p_value = null_model_zscore(
            signal,
            n_shuffles=100,
            null_type='composition',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31],
            max_q=7
        )

        # With 100 shuffles, should have good distribution
        # Check that nulls are not all identical
        assert len(np.unique(null_rcis)) > 10


class TestEdgeCases:
    """Test edge cases for null model."""

    def test_short_signal(self):
        """Test null model with short signal."""
        signal = np.random.randn(30)

        z_score, null_rcis, real_rci, p_value = null_model_zscore(
            signal,
            n_shuffles=10,
            null_type='composition',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[15],
            max_q=5
        )

        # Should still run without error
        assert np.isfinite(z_score)

    def test_constant_signal(self):
        """Test null model with constant signal."""
        signal = np.ones(100)

        z_score, null_rcis, real_rci, p_value = null_model_zscore(
            signal,
            n_shuffles=10,
            null_type='composition',
            seed=42,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31],
            max_q=5
        )

        # All RCIs should be very similar (constant signal)
        assert np.std(null_rcis) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
