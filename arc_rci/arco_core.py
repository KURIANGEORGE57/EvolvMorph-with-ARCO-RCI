"""
ARCO Core - Rational-frequency spectral fingerprinting and RCI computation

This module implements the core algorithms for:
1. Generating rational arcs (Farey sequence)
2. Computing power spectra from 1D signals
3. Integrating power around rational frequencies
4. Producing ARCO-print vectors and RCI scalars
5. Null model validation (composition-preserving and Markov-1 shuffles)

Frequency units: Arcs are expressed as cycles per sample (normalized frequency).
For sampled signals, multiply by sample_rate to get cycles/sec.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from fractions import Fraction
from scipy import signal as sp_signal
import warnings


def generate_rational_arcs(max_q: int = 11) -> List[Fraction]:
    """
    Generate sorted unique rational frequencies a/q using Farey sequence.

    Returns only canonical arcs where:
    - 1 <= q <= max_q
    - gcd(a, q) = 1 (coprime)
    - 0 < a/q < 0.5 (Nyquist mirror - only positive frequencies below Nyquist)

    Parameters
    ----------
    max_q : int
        Maximum denominator for rational arcs

    Returns
    -------
    List[Fraction]
        Sorted list of unique Fraction objects representing rational frequencies

    Examples
    --------
    >>> arcs = generate_rational_arcs(max_q=5)
    >>> [float(arc) for arc in arcs[:5]]
    [0.1, 0.14285714285714285, 0.16666666666666666, 0.2, 0.25]
    """
    arcs = set()

    # Generate all coprime rationals a/q for 1 <= q <= max_q
    for q in range(1, max_q + 1):
        for a in range(1, q):
            # Check coprimality using gcd
            if np.gcd(a, q) == 1:
                frac = Fraction(a, q)
                # Only keep frequencies below Nyquist (0.5)
                if float(frac) < 0.5:
                    arcs.add(frac)

    # Sort by value
    return sorted(list(arcs))


def sequence_to_track(signal_1d: np.ndarray, track_spec: Dict[str, Any]) -> np.ndarray:
    """
    Convert a raw 1D signal to one or more 'tracks' (e.g., amplitude, derivative, envelope).

    Parameters
    ----------
    signal_1d : np.ndarray
        Input 1D signal (shape: [N])
    track_spec : Dict[str, Any]
        Track specification with 'name' field. Supported names:
        - 'amp' or 'amplitude': returns signal as-is
        - 'deriv' or 'derivative': returns first derivative
        - 'deriv2' or 'second_derivative': returns second derivative
        - 'envelope': returns Hilbert envelope
        - 'diff': returns absolute difference

    Returns
    -------
    np.ndarray
        Processed track (shape: [N_out]). May be shorter due to differencing.

    Examples
    --------
    >>> sig = np.sin(2 * np.pi * 0.1 * np.arange(100))
    >>> track = sequence_to_track(sig, {'name': 'amp'})
    >>> track.shape
    (100,)
    """
    name = track_spec.get('name', 'amp').lower()

    if name in ['amp', 'amplitude']:
        return signal_1d

    elif name in ['deriv', 'derivative']:
        # First derivative (loses 1 sample)
        return np.diff(signal_1d)

    elif name in ['deriv2', 'second_derivative']:
        # Second derivative (loses 2 samples)
        return np.diff(signal_1d, n=2)

    elif name == 'envelope':
        # Hilbert envelope
        analytic = sp_signal.hilbert(signal_1d)
        return np.abs(analytic)

    elif name == 'diff':
        # Absolute difference
        return np.abs(np.diff(signal_1d))

    else:
        warnings.warn(f"Unknown track name '{name}', using amplitude")
        return signal_1d


def compute_power_spectrum(
    signal: np.ndarray,
    sample_rate: float,
    use_hann: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectrum using rfft.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (shape: [N])
    sample_rate : float
        Sampling rate (Hz for time-series, or 1.0 for sequences)
    use_hann : bool
        Whether to apply Hann window (corrects for energy loss)

    Returns
    -------
    freqs : np.ndarray
        Frequency bins (positive frequencies only, shape: [N//2 + 1])
    power : np.ndarray
        Normalized power spectrum (sum ~= 1.0)

    Notes
    -----
    - Uses rfft for real signals (returns positive frequencies only)
    - Applies Hann window if use_hann=True and corrects for energy loss
    - Power is normalized so total power = 1.0
    """
    N = len(signal)

    # Apply Hann window if requested
    if use_hann:
        window = np.hanning(N)
        windowed_signal = signal * window
        # Compute energy correction factor
        energy_correction = np.sum(window ** 2)
    else:
        windowed_signal = signal
        energy_correction = N

    # Compute FFT (real FFT for efficiency)
    fft_result = np.fft.rfft(windowed_signal)

    # Compute power (magnitude squared)
    power = np.abs(fft_result) ** 2

    # Correct for window energy loss
    power = power / energy_correction

    # Normalize so total power = 1
    total_power = np.sum(power)
    if total_power > 0:
        power = power / total_power

    # Compute frequency bins
    freqs = np.fft.rfftfreq(N, d=1.0/sample_rate)

    return freqs, power


def integrate_arc_power(
    freqs: np.ndarray,
    power: np.ndarray,
    arc: Fraction,
    q: int,
    width_scale: float = 1.0,
    scheme: str = 'gaussian'
) -> float:
    """
    Integrate power around a rational frequency with bandwidth ∝ 1/q².

    Parameters
    ----------
    freqs : np.ndarray
        Frequency bins from power spectrum
    power : np.ndarray
        Power values at each frequency bin
    arc : Fraction
        Rational frequency (as Fraction object)
    q : int
        Denominator of the rational frequency
    width_scale : float
        Bandwidth scaling factor (delta_factor)
    scheme : str
        Integration scheme: 'gaussian', 'hard', or 'triangular'

    Returns
    -------
    float
        Integrated power around the arc frequency

    Notes
    -----
    - Bandwidth sigma = width_scale / q² in normalized frequency units
    - For 'gaussian': uses Gaussian weighting exp(-0.5 * ((f - f0)/sigma)²)
    - For 'hard': rectangular window
    - For 'triangular': triangular window
    """
    # Get target frequency (normalized to [0, 0.5])
    f0 = float(arc)

    # Compute bandwidth: sigma = delta_factor / q^2
    # Normalized frequency units (cycles per sample)
    sigma = width_scale / (q ** 2)

    # Handle edge case
    if sigma <= 0:
        sigma = 1e-6

    # Normalize frequencies to [0, 0.5] range
    # freqs from rfftfreq are in absolute frequency units (Hz or cycles/sample depending on sample_rate)
    # Nyquist frequency = sample_rate / 2
    # We want to normalize to [0, 0.5] where 0.5 represents Nyquist
    # So freqs_normalized = freqs / sample_rate (this maps [0, sample_rate/2] to [0, 0.5])
    # But we need to infer sample_rate from freqs
    # For rfftfreq with N samples and d=1.0/sample_rate, max_freq = sample_rate/2
    max_freq = freqs[-1]  # This is sample_rate / 2 (Nyquist)
    if max_freq > 0:
        # Normalize to [0, 0.5]
        freqs_normalized = freqs / (2 * max_freq)  # Maps [0, max_freq] to [0, 0.5]
    else:
        freqs_normalized = freqs

    # Compute weights based on scheme
    if scheme == 'gaussian':
        # Gaussian window
        weights = np.exp(-0.5 * ((freqs_normalized - f0) / sigma) ** 2)

    elif scheme == 'hard':
        # Hard rectangular window (3-sigma cutoff)
        weights = np.where(np.abs(freqs_normalized - f0) <= 3 * sigma, 1.0, 0.0)

    elif scheme == 'triangular':
        # Triangular window
        dist = np.abs(freqs_normalized - f0)
        cutoff = 3 * sigma
        weights = np.maximum(0, 1 - dist / cutoff)

    else:
        raise ValueError(f"Unknown integration scheme: {scheme}")

    # Normalize weights
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum

    # Integrate power
    integrated = np.sum(power * weights)

    return integrated


def arco_from_signal(
    signal: np.ndarray,
    sample_rate: float,
    tracks: List[Dict],
    window_sizes: List[int] = [31, 63],
    step_fraction: float = 0.25,
    max_q: int = 11,
    delta_factor: float = 1.0,
    integration_scheme: str = 'gaussian',
    aggregation: str = 'mean'
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute multi-track ARCO fingerprint from a 1D signal.

    This is the main function that computes:
    1. ARCO-print: fixed-length feature vector
    2. RCI global: single scalar measure of rational periodicity
    3. RCI position: per-window RCI values

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal
    sample_rate : float
        Sampling rate (Hz for signals, 1.0 for sequences)
    tracks : List[Dict]
        List of track specifications, e.g., [{'name': 'amp'}, {'name': 'deriv'}]
    window_sizes : List[int]
        List of window sizes (in samples)
    step_fraction : float
        Step size as fraction of window (0.25 = 75% overlap)
    max_q : int
        Maximum denominator for rational arcs
    delta_factor : float
        Bandwidth scaling factor
    integration_scheme : str
        'gaussian', 'hard', or 'triangular'
    aggregation : str
        How to aggregate across windows: 'mean', 'max', 'concat'

    Returns
    -------
    arco_print : np.ndarray
        ARCO fingerprint vector (shape: [D])
    rci_global : float
        Global RCI value (mean of window RCIs)
    rci_position : np.ndarray
        Per-window RCI values (shape: [n_windows])

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.1 * np.arange(200))
    >>> arco, rci, rci_pos = arco_from_signal(
    ...     signal, sample_rate=1.0,
    ...     tracks=[{'name': 'amp'}],
    ...     window_sizes=[50],
    ...     max_q=11
    ... )
    >>> arco.shape[0] > 0
    True
    """
    # Generate rational arcs
    rational_arcs = generate_rational_arcs(max_q)
    n_arcs = len(rational_arcs)

    # Storage for features across all windows and window sizes
    all_window_features = []
    all_window_rcis = []

    # Process each window size
    for win_size in window_sizes:
        if win_size > len(signal):
            warnings.warn(f"Window size {win_size} > signal length {len(signal)}, using whole signal")
            win_size = len(signal)

        # Compute step size
        step = max(1, int(win_size * step_fraction))

        # Generate windows
        n_windows = (len(signal) - win_size) // step + 1

        if n_windows <= 0:
            n_windows = 1
            windows = [signal]
        else:
            windows = [signal[i*step : i*step + win_size] for i in range(n_windows)]

        # Process each window
        for window_data in windows:
            # Storage for this window across tracks
            window_arc_powers = []

            # Process each track
            for track_spec in tracks:
                # Convert to track
                track_signal = sequence_to_track(window_data, track_spec)

                # Skip if too short
                if len(track_signal) < 3:
                    continue

                # Compute power spectrum
                freqs, power = compute_power_spectrum(track_signal, sample_rate, use_hann=True)

                # Integrate power at each rational arc
                arc_powers = []
                for arc in rational_arcs:
                    q = arc.denominator
                    integrated = integrate_arc_power(
                        freqs, power, arc, q,
                        width_scale=delta_factor,
                        scheme=integration_scheme
                    )
                    arc_powers.append(integrated)

                window_arc_powers.append(arc_powers)

            # Average across tracks for this window
            if window_arc_powers:
                mean_arc_powers = np.mean(window_arc_powers, axis=0)
                all_window_features.append(mean_arc_powers)

                # Compute RCI for this window (sum of rational power)
                rci_window = np.sum(mean_arc_powers)
                all_window_rcis.append(rci_window)

    # Aggregate across windows
    if not all_window_features:
        # Return zeros if no valid windows
        return (
            np.zeros(n_arcs),
            0.0,
            np.array([0.0])
        )

    all_window_features = np.array(all_window_features)
    all_window_rcis = np.array(all_window_rcis)

    # Aggregate features
    if aggregation == 'mean':
        arco_print = np.mean(all_window_features, axis=0)
    elif aggregation == 'max':
        arco_print = np.max(all_window_features, axis=0)
    elif aggregation == 'concat':
        arco_print = all_window_features.flatten()
    else:
        arco_print = np.mean(all_window_features, axis=0)

    # Global RCI (mean of window RCIs)
    rci_global = np.mean(all_window_rcis)

    return arco_print, rci_global, all_window_rcis


def null_model_zscore(
    signal: np.ndarray,
    n_shuffles: int = 50,
    null_type: str = 'composition',
    seed: Optional[int] = None,
    **arco_kwargs
) -> Tuple[float, np.ndarray, float, float]:
    """
    Compute z-score of real RCI vs. shuffled null model.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    n_shuffles : int
        Number of shuffles for null distribution
    null_type : str
        'composition' for random permutation, 'markov1' for Markov-1 preserving
    seed : Optional[int]
        Random seed for reproducibility
    **arco_kwargs
        Arguments passed to arco_from_signal

    Returns
    -------
    z_score : float
        Z-score of real RCI compared to null distribution
    null_rcis : np.ndarray
        RCI values from shuffled signals
    real_rci : float
        RCI of original signal
    p_value : float
        Empirical p-value (fraction of null >= real)

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.1 * np.arange(200))
    >>> z, nulls, real, pval = null_model_zscore(
    ...     signal, n_shuffles=20,
    ...     sample_rate=1.0, tracks=[{'name': 'amp'}],
    ...     window_sizes=[50], max_q=11
    ... )
    >>> z > 0  # Periodic signal should have positive z-score
    True
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute RCI for real signal
    _, real_rci, _ = arco_from_signal(signal, **arco_kwargs)

    # Compute null distribution
    null_rcis = []

    for _ in range(n_shuffles):
        if null_type == 'composition':
            # Random permutation (preserves composition)
            shuffled = np.random.permutation(signal)

        elif null_type == 'markov1':
            # Markov-1 shuffle (preserves first-order transitions)
            shuffled = _markov1_shuffle(signal)

        else:
            raise ValueError(f"Unknown null_type: {null_type}")

        # Compute RCI for shuffled
        _, rci_null, _ = arco_from_signal(shuffled, **arco_kwargs)
        null_rcis.append(rci_null)

    null_rcis = np.array(null_rcis)

    # Compute z-score
    null_mean = np.mean(null_rcis)
    null_std = np.std(null_rcis)

    if null_std > 0:
        z_score = (real_rci - null_mean) / null_std
    else:
        z_score = 0.0

    # Compute p-value (one-tailed: fraction of null >= real)
    p_value = np.sum(null_rcis >= real_rci) / n_shuffles

    return z_score, null_rcis, real_rci, p_value


def _markov1_shuffle(signal: np.ndarray) -> np.ndarray:
    """
    Generate a Markov-1 shuffle that preserves first-order transition probabilities.

    Uses Euler path approach to construct a sequence with same transition matrix.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Shuffled signal with same transition statistics
    """
    N = len(signal)
    if N <= 1:
        return signal.copy()

    # Build transition graph
    from collections import defaultdict, deque

    # Create edges (transitions)
    edges = defaultdict(list)
    for i in range(N - 1):
        edges[signal[i]].append(signal[i + 1])

    # Shuffle edges for each state
    for state in edges:
        np.random.shuffle(edges[state])

    # Build new sequence by following transitions
    # Start with first element
    shuffled = [signal[0]]
    edge_counts = {k: 0 for k in edges.keys()}

    for _ in range(N - 1):
        current = shuffled[-1]
        if current in edges and edge_counts[current] < len(edges[current]):
            next_val = edges[current][edge_counts[current]]
            edge_counts[current] += 1
            shuffled.append(next_val)
        else:
            # If we run out of edges, fall back to random choice
            shuffled.append(np.random.choice(signal))

    return np.array(shuffled)
