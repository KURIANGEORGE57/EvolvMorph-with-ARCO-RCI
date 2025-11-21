# ARCO/RCI Module

**Rational-frequency spectral fingerprinting for 1D signals and sequences**

This module computes multitrack rational-frequency fingerprints (ARCO-print), sliding-window ARCO-3D maps, and per-sample/global RCI scalars. It is designed for spectral feature extraction from 1D signals, sequences, and XRD patterns.

---

## Features

- **ARCO Fingerprints**: Fixed-length feature vectors based on rational-frequency power integration
- **RCI Scalars**: Rational Complexity Index - measures periodicity strength
- **Multi-track Analysis**: Analyze amplitude, derivatives, envelopes, etc.
- **Null Model Validation**: Composition-preserving and Markov-1 shuffles for significance testing
- **Visualization**: Arcograms (heatmaps) and polar arcograms
- **CLI Interface**: Batch processing from command line
- **Python API**: Easy integration into pipelines

---

## Installation

### Dependencies

```bash
pip install numpy scipy matplotlib
```

For testing:
```bash
pip install pytest
```

### Quick Install

The module is self-contained and can be used directly:

```bash
cd arc_rci
python -m pytest tests/  # Run tests
```

---

## Quick Start

### Python API

```python
from arc_rci import arco_from_signal, generate_rational_arcs
import numpy as np

# Create or load a signal
signal = np.sin(2 * np.pi * 0.1 * np.arange(200))

# Compute ARCO fingerprint
arco_print, rci_global, rci_position = arco_from_signal(
    signal,
    sample_rate=1.0,              # 1.0 for sequences, Hz for sampled signals
    tracks=[{'name': 'amp'}],     # Track types
    window_sizes=[31, 63],        # Window sizes
    max_q=11,                     # Maximum denominator for rational arcs
    delta_factor=1.0              # Bandwidth scaling
)

print(f"ARCO fingerprint shape: {arco_print.shape}")
print(f"Global RCI: {rci_global:.4f}")
```

### Command-Line Interface

```bash
# Process all .npy files in a directory
python -m arc_rci.arco_cli \
    --input signals/ \
    --pattern "*.npy" \
    --out features/ \
    --fs 500 \
    --max-q 15 \
    --window-sizes 31 63 \
    --step-fraction 0.25

# Single file with visualization
python -m arc_rci.arco_cli \
    --input signal.npy \
    --out output/ \
    --fs 1.0 \
    --visualize \
    --viz-type arcogram polar quick

# With null model validation
python -m arc_rci.arco_cli \
    --input signal.npy \
    --out output/ \
    --null-model \
    --n-shuffles 100
```

---

## Core Concepts

### Rational Arcs

ARCO uses **rational frequencies** (fractions a/q where gcd(a,q)=1) as basis frequencies:
- Example arcs: 1/2, 1/3, 2/3, 1/4, 3/4, 1/5, 2/5, 3/5, 4/5, 1/6, 5/6, 1/7, ...
- Covers all periodic structures up to denominator `max_q`
- More interpretable than arbitrary Fourier frequencies

### ARCO Fingerprint

For each window and track:
1. Compute power spectrum (FFT)
2. Integrate power around each rational frequency using Gaussian windows (bandwidth ∝ 1/q²)
3. Aggregate across windows → fixed-length ARCO-print vector

**Dimension**: `n_tracks × n_arcs` (or flattened)

### RCI (Rational Complexity Index)

**RCI = sum of integrated power at all rational frequencies**

- High RCI → strong rational periodicity
- Low RCI → weak or non-rational periodicity
- Range: typically [0, 1], but can exceed 1 depending on normalization

### Null Models

To assess significance, compare real RCI to shuffled null distributions:

1. **Composition-preserving**: Random permutation (preserves residue/sample composition)
2. **Markov-1**: Preserves first-order transition probabilities

**Z-score > 3.0** → statistically significant periodicity (p < 0.003)

---

## Parameters Guide

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 1.0 | Sampling rate (Hz for signals, 1.0 for sequences) |
| `max_q` | 11 | Maximum denominator for rational arcs (11 for proteins, 30+ for signals) |
| `window_sizes` | [31, 63] | Window sizes in samples (use multiple scales) |
| `step_fraction` | 0.25 | Step size as fraction of window (0.25 = 75% overlap) |
| `delta_factor` | 1.0 | Bandwidth scaling (bandwidth = delta_factor / q²) |
| `integration_scheme` | 'gaussian' | Integration scheme: 'gaussian', 'hard', or 'triangular' |

### Track Types

- `amp` / `amplitude`: Raw signal (identity)
- `deriv` / `derivative`: First derivative
- `deriv2`: Second derivative
- `envelope`: Hilbert envelope
- `diff`: Absolute difference

### Recommended Settings

**Protein sequences**:
```python
sample_rate=1.0, max_q=11, window_sizes=[31, 63], tracks=['amp', 'hydrophobicity']
```

**ECG/physiological signals** (500 Hz):
```python
sample_rate=500, max_q=30, window_sizes=[250, 500], tracks=['amp', 'deriv']
```

**XRD patterns**:
```python
sample_rate=1.0, max_q=15, window_sizes=[50, 100], tracks=['amp']
```

---

## Examples

### Example 1: Detect 7-mer Periodicity

```python
from arc_rci import arco_from_signal, generate_rational_arcs
from fractions import Fraction
import numpy as np

# Create perfect 7-mer repeat
motif = np.random.randn(7)
signal = np.tile(motif, 30)

# Compute ARCO
arco, rci, rci_pos = arco_from_signal(
    signal,
    sample_rate=1.0,
    tracks=[{'name': 'amp'}],
    window_sizes=[35, 70],  # Multiples of 7
    max_q=11
)

# Find top arcs
arcs = generate_rational_arcs(max_q=11)
top_idx = np.argmax(arco)
top_arc = arcs[top_idx]

print(f"Top arc: {top_arc} (expected: 1/7)")
print(f"RCI: {rci:.4f}")
```

### Example 2: Null Model Validation

```python
from arc_rci import null_model_zscore
import numpy as np

signal = np.sin(2 * np.pi * 0.1 * np.arange(200))

z_score, null_rcis, real_rci, p_value = null_model_zscore(
    signal,
    n_shuffles=50,
    null_type='composition',
    seed=42,
    sample_rate=1.0,
    tracks=[{'name': 'amp'}],
    window_sizes=[50],
    max_q=11
)

print(f"Real RCI: {real_rci:.4f}")
print(f"Z-score: {z_score:.2f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant: {z_score > 3.0}")
```

### Example 3: Visualization

```python
from arc_rci import arco_from_signal, generate_rational_arcs
from arc_rci.arco_vis import plot_arcogram, plot_polar_arcogram, quick_visualize
import numpy as np

signal = np.sin(2 * np.pi * 0.1 * np.arange(200))

arco, rci, rci_pos = arco_from_signal(
    signal,
    sample_rate=1.0,
    tracks=[{'name': 'amp'}, {'name': 'deriv'}],
    window_sizes=[50],
    max_q=11
)

arcs = generate_rational_arcs(max_q=11)

# Arcogram (heatmap)
plot_arcogram(
    arco.reshape(2, -1),  # 2 tracks
    arcs,
    track_names=['Amplitude', 'Derivative'],
    out_file='arcogram.png'
)

# Polar arcogram
plot_polar_arcogram(
    arco.mean(axis=0) if arco.ndim == 2 else arco,
    arcs,
    out_file='polar.png'
)

# Quick comprehensive view
quick_visualize(
    signal, arco, arcs, rci, rci_pos,
    track_names=['Amplitude', 'Derivative'],
    out_file='quick.png'
)
```

### Example 4: Batch Processing

```python
from arc_rci.arco_io import batch_load_signals, save_batch_arco
from arc_rci import arco_from_signal, generate_rational_arcs

# Load all signals
signals = batch_load_signals('data/', pattern='*.npy')

# Process each
results = {}
arcs = generate_rational_arcs(max_q=11)

for name, signal in signals.items():
    arco, rci, rci_pos = arco_from_signal(
        signal,
        sample_rate=1.0,
        tracks=[{'name': 'amp'}],
        window_sizes=[31, 63],
        max_q=11
    )

    results[name] = {
        'arco_print': arco,
        'rci_global': rci,
        'rci_position': rci_pos,
        'rational_list': arcs
    }

# Save all
save_batch_arco(results, 'output/', format='npz')
```

---

## Integration with EvolvMorph

### As Feature Extractor

```python
# In your EvolvMorph pipeline:
from arc_rci import arco_from_signal

# For each XRD pattern or signal:
xrd_intensities = load_xrd_pattern(...)  # Your existing loader

# Extract ARCO features
arco_features, rci, _ = arco_from_signal(
    xrd_intensities,
    sample_rate=1.0,
    tracks=[{'name': 'amp'}],
    window_sizes=[50, 100],
    max_q=15
)

# Add to feature database
features = np.concatenate([other_features, arco_features])
```

### In Bayesian Optimization

```python
# Add ARCO as surrogate model input:
X_train_with_arco = np.hstack([X_train, arco_features])
```

---

## File Formats

### Input Formats

- `.npy`: NumPy array (1D)
- `.npz`: NumPy archive (uses first array or key 'signal'/'data')
- `.fasta`: FASTA sequence files
- `.csv`: CSV with 2 columns (angle, intensity) for XRD

### Output Formats

#### NPZ Format
```python
np.load('output_ARCO.npz')
# Keys:
#   'arco_print': ARCO fingerprint vector
#   'rci_global': global RCI scalar
#   'rci_position': per-window RCI array
#   'rational_list': list of rational arc strings
#   'track_names': track names
#   'metadata': JSON metadata
```

#### CSV Format
```
# metadata
1/10,1/9,1/8,1/7,...
0.0123,0.0234,0.0345,...
```

---

## Testing

Run unit tests:

```bash
cd arc_rci
python -m pytest tests/ -v
```

Tests include:
- Rational arc generation
- Power spectrum computation
- Detection of known periodicities (7-mer, 3-mer)
- Null model validation
- Edge cases

---

## Performance

- **Time complexity**: O(K × T × N log N) per signal
  - K = number of tracks
  - T = number of windows
  - N = window size

- **Memory**: ARCO vector dimension ≈ n_tracks × n_arcs × n_window_sizes
  - Example: 2 tracks × 42 arcs × 2 windows = 168 floats (~1 KB)

- **Typical timing** (on modern CPU):
  - 200-sample signal, max_q=11: ~50 ms
  - 1000-sample signal, max_q=30: ~200 ms

---

## API Reference

### Core Functions

#### `arco_from_signal(signal, sample_rate, tracks, window_sizes, ...)`
Compute ARCO fingerprint from 1D signal.

**Returns**: `(arco_print, rci_global, rci_position)`

#### `generate_rational_arcs(max_q)`
Generate list of rational frequency arcs.

**Returns**: `List[Fraction]`

#### `null_model_zscore(signal, n_shuffles, null_type, ...)`
Compute z-score vs. null model.

**Returns**: `(z_score, null_rcis, real_rci, p_value)`

### I/O Functions

- `load_signal_from_numpy(path)`: Load .npy/.npz
- `save_arco_npz(arco_vector, out_path, ...)`: Save to .npz
- `batch_load_signals(input_dir, pattern)`: Batch load
- `save_batch_arco(results, output_dir, format)`: Batch save

### Visualization

- `plot_arcogram(...)`: Heatmap of tracks × rationals
- `plot_polar_arcogram(...)`: Polar plot
- `plot_rci_position(...)`: RCI position profile
- `quick_visualize(...)`: Comprehensive multi-panel plot

---

## Citation

If you use this ARCO/RCI module in your research, please cite:

```
@software{arco_rci_2024,
  title = {ARCO/RCI: Rational-frequency spectral fingerprinting},
  author = {Claude Code},
  year = {2024},
  note = {Module for EvolvMorph-with-ARCO-RCI}
}
```

And the parent EvolvMorph project:

```
Joohwi Lee, Junpei Oba, Nobuko Ohba, and Seiji Kajita,
npj Computational Materials, 9, 135 (2023).
```

---

## License

Same license as EvolvMorph (Toyota Central R&D Labs., Inc.)

For non-commercial research purposes only.

---

## Support

For issues or questions:
- Open an issue in the repository
- Contact the EvolvMorph team

---

## Version History

- **v1.0.0** (2024): Initial release
  - Core ARCO/RCI algorithms
  - Multi-track support
  - Null model validation
  - CLI and Python API
  - Visualization tools
