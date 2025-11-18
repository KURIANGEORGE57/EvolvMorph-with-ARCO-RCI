"""
ARCO/RCI Module - Rational-frequency spectral fingerprinting

A self-contained module for computing multitrack rational-frequency fingerprints
(ARCO-print), sliding-window ARCO-3D maps, and per-sample/global RCI scalars.

Main components:
- arco_core: Core algorithms for ARCO computation and null models
- arco_io: Input/output utilities
- arco_vis: Visualization tools (arcograms, polar arcograms)
- arco_cli: Command-line interface

Quick start:
    >>> from arc_rci import arco_from_signal, generate_rational_arcs
    >>> import numpy as np
    >>> signal = np.sin(2 * np.pi * 0.1 * np.arange(200))
    >>> arco, rci, rci_pos = arco_from_signal(
    ...     signal,
    ...     sample_rate=1.0,
    ...     tracks=[{'name': 'amp'}],
    ...     window_sizes=[50],
    ...     max_q=11
    ... )

Authors: Claude Code (Implementation based on specifications)
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Claude Code'

# Core functions
from .arco_core import (
    generate_rational_arcs,
    sequence_to_track,
    compute_power_spectrum,
    integrate_arc_power,
    arco_from_signal,
    null_model_zscore
)

# I/O functions
from .arco_io import (
    load_signal_from_numpy,
    load_sequence_from_fasta,
    sequence_to_numeric,
    save_arco_npz,
    save_arco_csv,
    load_arco_npz,
    load_xrd_spectrum,
    batch_load_signals,
    save_batch_arco
)

# Visualization functions
from .arco_vis import (
    plot_arcogram,
    plot_polar_arcogram,
    plot_rci_position,
    plot_null_distribution,
    quick_visualize
)

# Define public API
__all__ = [
    # Core
    'generate_rational_arcs',
    'sequence_to_track',
    'compute_power_spectrum',
    'integrate_arc_power',
    'arco_from_signal',
    'null_model_zscore',

    # I/O
    'load_signal_from_numpy',
    'load_sequence_from_fasta',
    'sequence_to_numeric',
    'save_arco_npz',
    'save_arco_csv',
    'load_arco_npz',
    'load_xrd_spectrum',
    'batch_load_signals',
    'save_batch_arco',

    # Visualization
    'plot_arcogram',
    'plot_polar_arcogram',
    'plot_rci_position',
    'plot_null_distribution',
    'quick_visualize',
]
