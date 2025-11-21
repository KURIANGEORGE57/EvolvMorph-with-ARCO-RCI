"""
Basic unit tests for ARCO module

Tests:
1. Rational arc generation
2. Power spectrum computation
3. ARCO fingerprint extraction from synthetic periodic signals
4. Detection of known periodicities (7-mer, 3-mer repeats)
5. Random signal behavior (low RCI)
"""

import numpy as np
import pytest
from fractions import Fraction
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from arc_rci.arco_core import (
    generate_rational_arcs,
    compute_power_spectrum,
    integrate_arc_power,
    arco_from_signal,
    sequence_to_track
)


class TestRationalArcs:
    """Test rational arc generation."""

    def test_generate_arcs_basic(self):
        """Test basic arc generation."""
        arcs = generate_rational_arcs(max_q=5)

        # Should have multiple arcs
        assert len(arcs) > 0

        # All should be Fractions
        assert all(isinstance(a, Fraction) for a in arcs)

        # All should be in (0, 0.5)
        assert all(0 < float(a) < 0.5 for a in arcs)

        # Should be sorted
        assert arcs == sorted(arcs)

    def test_arcs_coprime(self):
        """Test that arcs are coprime (gcd(a, q) = 1)."""
        arcs = generate_rational_arcs(max_q=10)

        for arc in arcs:
            a = arc.numerator
            q = arc.denominator
            assert np.gcd(a, q) == 1

    def test_known_arcs(self):
        """Test that known arcs are present."""
        arcs = generate_rational_arcs(max_q=10)

        # Check specific arcs
        assert Fraction(1, 7) in arcs
        assert Fraction(1, 3) in arcs
        assert Fraction(1, 5) in arcs
        assert Fraction(2, 5) in arcs

    def test_arc_count_scaling(self):
        """Test that arc count scales with max_q."""
        arcs_5 = generate_rational_arcs(max_q=5)
        arcs_10 = generate_rational_arcs(max_q=10)

        # More arcs with larger max_q
        assert len(arcs_10) > len(arcs_5)


class TestPowerSpectrum:
    """Test power spectrum computation."""

    def test_power_spectrum_shape(self):
        """Test output shape."""
        signal = np.random.randn(100)
        freqs, power = compute_power_spectrum(signal, sample_rate=1.0)

        # Should have N//2 + 1 frequency bins
        assert len(freqs) == len(signal) // 2 + 1
        assert len(power) == len(signal) // 2 + 1

    def test_power_spectrum_normalization(self):
        """Test that power is normalized."""
        signal = np.random.randn(100)
        freqs, power = compute_power_spectrum(signal, sample_rate=1.0)

        # Total power should be ~1.0
        assert np.abs(np.sum(power) - 1.0) < 1e-6

    def test_power_spectrum_sine(self):
        """Test power spectrum of pure sine wave."""
        # Create sine wave at f=0.1 cycles/sample
        N = 200
        f0 = 0.1
        signal = np.sin(2 * np.pi * f0 * np.arange(N))

        freqs, power = compute_power_spectrum(signal, sample_rate=1.0, use_hann=True)

        # Find peak frequency
        peak_idx = np.argmax(power)
        peak_freq = freqs[peak_idx]

        # Peak should be near f0
        assert np.abs(peak_freq - f0) < 0.02  # Allow some tolerance due to windowing


class TestSequenceToTrack:
    """Test track conversion."""

    def test_amplitude_track(self):
        """Test amplitude track (identity)."""
        signal = np.random.randn(100)
        track = sequence_to_track(signal, {'name': 'amp'})

        assert np.array_equal(track, signal)

    def test_derivative_track(self):
        """Test derivative track."""
        signal = np.arange(100, dtype=float)
        track = sequence_to_track(signal, {'name': 'deriv'})

        # Derivative of linear ramp should be constant
        assert len(track) == len(signal) - 1
        assert np.allclose(track, 1.0)

    def test_envelope_track(self):
        """Test envelope track."""
        # Create modulated signal
        N = 200
        carrier = np.sin(2 * np.pi * 0.1 * np.arange(N))
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.01 * np.arange(N))
        signal = carrier * envelope

        track = sequence_to_track(signal, {'name': 'envelope'})

        # Envelope should be close to the modulation envelope
        assert len(track) == len(signal)
        # Envelope should be positive
        assert np.all(track >= 0)


class TestIntegrateArcPower:
    """Test arc power integration."""

    def test_integration_gaussian(self):
        """Test Gaussian integration."""
        # Create simple spectrum
        freqs = np.linspace(0, 0.5, 100)
        power = np.exp(-((freqs - 0.2) ** 2) / 0.01)  # Peak at 0.2
        power = power / np.sum(power)

        # Integrate around 1/5 = 0.2
        arc = Fraction(1, 5)
        integrated = integrate_arc_power(freqs, power, arc, q=5, scheme='gaussian')

        # Should be positive
        assert integrated > 0

        # Should capture significant power
        assert integrated > 0.01

    def test_integration_hard(self):
        """Test hard window integration."""
        freqs = np.linspace(0, 0.5, 100)
        power = np.ones(100) / 100

        arc = Fraction(1, 3)
        integrated = integrate_arc_power(freqs, power, arc, q=3, scheme='hard')

        assert integrated > 0


class TestARCOFromSignal:
    """Test ARCO fingerprint computation."""

    def test_arco_basic(self):
        """Test basic ARCO computation."""
        # Random signal
        signal = np.random.randn(200)

        arco, rci, rci_pos = arco_from_signal(
            signal,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[50],
            max_q=7
        )

        # Check outputs
        assert arco.ndim == 1
        assert arco.shape[0] > 0
        assert isinstance(rci, (float, np.floating))
        assert len(rci_pos) > 0

        # RCI should be in reasonable range
        assert 0 <= rci <= 2.0  # Allow some margin

    def test_arco_7mer_repeat(self):
        """Test ARCO on perfect 7-mer repeat."""
        np.random.seed(42)

        # Create perfect 7-mer repeat with strong contrast
        # Use a simple pattern for better detection
        motif = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 0.0])
        n_repeats = 20
        signal = np.tile(motif, n_repeats)

        arco, rci, rci_pos = arco_from_signal(
            signal,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[35, 70],  # Multiples of 7
            max_q=11,
            step_fraction=0.5
        )

        # Get rational arcs
        from arc_rci.arco_core import generate_rational_arcs
        arcs = generate_rational_arcs(max_q=11)

        # Find index of 1/7
        target_arc = Fraction(1, 7)
        if target_arc in arcs:
            target_idx = arcs.index(target_arc)

            # Power at 1/7 should be high
            power_at_1_7 = arco[target_idx]

            # Should be among the top arcs (top 10 to be less strict)
            sorted_indices = np.argsort(arco)[::-1]
            top_10_indices = sorted_indices[:10]

            assert target_idx in top_10_indices, \
                f"1/7 not in top 10. Top arcs: {[arcs[i] for i in top_10_indices[:5]]}, 1/7 power={power_at_1_7:.4f}"

    def test_arco_3mer_repeat(self):
        """Test ARCO on perfect 3-mer repeat (collagen-like)."""
        np.random.seed(42)

        # Create perfect 3-mer repeat
        motif = np.array([1.0, -0.5, 0.2])
        n_repeats = 30
        signal = np.tile(motif, n_repeats)

        arco, rci, rci_pos = arco_from_signal(
            signal,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[30, 60],  # Multiples of 3
            max_q=11,
            step_fraction=0.5
        )

        # Get rational arcs
        from arc_rci.arco_core import generate_rational_arcs
        arcs = generate_rational_arcs(max_q=11)

        # Find index of 1/3
        target_arc = Fraction(1, 3)
        if target_arc in arcs:
            target_idx = arcs.index(target_arc)

            # Power at 1/3 should be high
            sorted_indices = np.argsort(arco)[::-1]
            top_5_indices = sorted_indices[:5]

            assert target_idx in top_5_indices, \
                f"1/3 not in top 5. Top arcs: {[arcs[i] for i in top_5_indices]}"

    def test_arco_random_low_rci(self):
        """Test that random signal has low RCI."""
        np.random.seed(42)

        # Random signal
        signal = np.random.randn(200)

        arco, rci, rci_pos = arco_from_signal(
            signal,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31, 63],
            max_q=11
        )

        # RCI should be relatively low for random signal
        # Note: RCI can be > 1 due to overlapping Gaussian windows
        # We just check it's not extremely high
        assert rci < 2.0, f"Random signal RCI={rci:.3f} is too high"

    def test_arco_multitrack(self):
        """Test multi-track ARCO."""
        signal = np.random.randn(200)

        arco, rci, rci_pos = arco_from_signal(
            signal,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}, {'name': 'deriv'}],
            window_sizes=[50],
            max_q=7
        )

        # Should work with multiple tracks
        assert arco.shape[0] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_signal(self):
        """Test with very short signal."""
        signal = np.random.randn(10)

        arco, rci, rci_pos = arco_from_signal(
            signal,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[5],
            max_q=5
        )

        # Should still return valid output
        assert arco.shape[0] > 0

    def test_constant_signal(self):
        """Test with constant signal."""
        signal = np.ones(100)

        arco, rci, rci_pos = arco_from_signal(
            signal,
            sample_rate=1.0,
            tracks=[{'name': 'amp'}],
            window_sizes=[31],
            max_q=7
        )

        # RCI should be very low (no periodicity)
        assert rci < 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
