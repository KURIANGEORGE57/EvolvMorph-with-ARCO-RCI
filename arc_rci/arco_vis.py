"""
ARCO Visualization - Plotting utilities for ARCO fingerprints

Provides functions to:
1. Plot arcograms (heatmap of tracks × rational frequencies)
2. Plot polar arcograms (polar plot of arc powers)
3. Quick visualization utilities
"""

import numpy as np
from typing import List, Optional
from fractions import Fraction
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


def plot_arcogram(
    arco_matrix: np.ndarray,
    rational_list: List[Fraction],
    track_names: Optional[List[str]] = None,
    out_file: Optional[str] = None,
    title: str = "ARCO-gram",
    cmap: str = 'viridis',
    figsize: tuple = (12, 6),
    show_values: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
):
    """
    Plot arcogram heatmap (tracks × rational frequencies).

    Parameters
    ----------
    arco_matrix : np.ndarray
        ARCO matrix (shape: [n_tracks, n_arcs] or [n_arcs] for single track)
    rational_list : List[Fraction]
        List of rational arc frequencies
    track_names : List[str], optional
        Names of tracks
    out_file : str, optional
        Output file path (if None, displays plot)
    title : str
        Plot title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    show_values : bool
        Whether to show numeric values in cells
    vmin, vmax : float, optional
        Color scale limits

    Examples
    --------
    >>> from fractions import Fraction
    >>> arco = np.random.rand(5, 10)
    >>> rats = [Fraction(i, 10) for i in range(1, 11)]
    >>> # plot_arcogram(arco, rats, track_names=['amp', 'deriv'])
    """
    # Handle 1D input
    if arco_matrix.ndim == 1:
        arco_matrix = arco_matrix.reshape(1, -1)

    n_tracks, n_arcs = arco_matrix.shape

    # Create labels
    if track_names is None:
        track_names = [f"Track {i+1}" for i in range(n_tracks)]

    # Create rational labels (show fraction and decimal)
    arc_labels = [f"{r}\n({float(r):.3f})" for r in rational_list[:n_arcs]]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(
        arco_matrix,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(n_arcs))
    ax.set_yticks(np.arange(n_tracks))

    # Labels
    ax.set_xticklabels(arc_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(track_names, fontsize=10)

    ax.set_xlabel('Rational Frequency (cycles/sample)', fontsize=12)
    ax.set_ylabel('Track', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Integrated Power', fontsize=10)

    # Optionally show values
    if show_values and n_arcs <= 20:
        for i in range(n_tracks):
            for j in range(n_arcs):
                text = ax.text(
                    j, i, f'{arco_matrix[i, j]:.3f}',
                    ha="center", va="center",
                    color="white" if arco_matrix[i, j] > (arco_matrix.max() / 2) else "black",
                    fontsize=6
                )

    # Grid
    ax.set_xticks(np.arange(n_arcs + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_tracks + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_polar_arcogram(
    arco_vector: np.ndarray,
    rational_list: List[Fraction],
    out_file: Optional[str] = None,
    title: str = "Polar ARCO-gram",
    figsize: tuple = (10, 10),
    cmap: str = 'plasma',
    radial_scale: str = 'linear',
    show_labels: bool = True
):
    """
    Plot polar arcogram (arcs arranged by angle, radius = power).

    Rational frequencies are mapped to polar coordinates:
    - Angle: proportional to frequency value
    - Radius: integrated power at that frequency

    Parameters
    ----------
    arco_vector : np.ndarray
        ARCO vector (1D, shape: [n_arcs])
    rational_list : List[Fraction]
        List of rational arc frequencies
    out_file : str, optional
        Output file path
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap for coloring arcs by denominator
    radial_scale : str
        'linear' or 'log' for radius scaling
    show_labels : bool
        Whether to show fraction labels

    Examples
    --------
    >>> from fractions import Fraction
    >>> arco = np.random.rand(10)
    >>> rats = [Fraction(i, 10) for i in range(1, 11)]
    >>> # plot_polar_arcogram(arco, rats)
    """
    n_arcs = len(arco_vector)

    # Create figure with polar projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')

    # Map frequencies to angles (0 to π, since we only use 0 to 0.5)
    freqs = np.array([float(r) for r in rational_list[:n_arcs]])
    # Map [0, 0.5] to [0, π]
    theta = freqs * 2 * np.pi

    # Get denominators for coloring
    denominators = np.array([r.denominator for r in rational_list[:n_arcs]])

    # Radius is power
    radius = arco_vector

    # Apply radial scaling
    if radial_scale == 'log':
        radius = np.log1p(radius)

    # Normalize denominators for color mapping
    norm = mpl.colors.Normalize(vmin=denominators.min(), vmax=denominators.max())
    cmap_obj = plt.get_cmap(cmap)

    # Plot each arc
    for i in range(n_arcs):
        color = cmap_obj(norm(denominators[i]))

        # Plot as a wedge/bar
        ax.plot(
            [theta[i], theta[i]],
            [0, radius[i]],
            color=color,
            linewidth=2,
            alpha=0.7
        )

        # Add marker at tip
        ax.plot(
            theta[i], radius[i],
            'o',
            color=color,
            markersize=8,
            alpha=0.9
        )

        # Add label
        if show_labels and n_arcs <= 30:
            # Position label slightly outside
            label_radius = radius[i] * 1.1
            ax.text(
                theta[i], label_radius,
                str(rational_list[i]),
                ha='center', va='center',
                fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )

    # Configure polar plot
    ax.set_theta_zero_location('N')  # 0° at top
    ax.set_theta_direction(1)  # Clockwise

    # Set radial limits
    ax.set_ylim(0, radius.max() * 1.2)

    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add colorbar for denominators
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Denominator (q)', fontsize=10)

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_rci_position(
    rci_position: np.ndarray,
    out_file: Optional[str] = None,
    title: str = "RCI Position Profile",
    figsize: tuple = (12, 4),
    threshold: Optional[float] = None
):
    """
    Plot RCI values along sequence position (windowed RCI).

    Parameters
    ----------
    rci_position : np.ndarray
        Per-window RCI values
    out_file : str, optional
        Output file path
    title : str
        Plot title
    figsize : tuple
        Figure size
    threshold : float, optional
        Significance threshold line to draw

    Examples
    --------
    >>> rci_pos = np.random.rand(50)
    >>> # plot_rci_position(rci_pos, threshold=0.3)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot RCI profile
    x = np.arange(len(rci_position))
    ax.plot(x, rci_position, linewidth=2, color='steelblue', label='RCI')
    ax.fill_between(x, 0, rci_position, alpha=0.3, color='steelblue')

    # Add threshold line
    if threshold is not None:
        ax.axhline(
            threshold, color='red', linestyle='--',
            linewidth=1.5, label=f'Threshold = {threshold:.2f}'
        )

    # Labels
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('RCI Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_null_distribution(
    null_rcis: np.ndarray,
    real_rci: float,
    z_score: float,
    out_file: Optional[str] = None,
    title: str = "Null Model Distribution",
    figsize: tuple = (10, 6)
):
    """
    Plot null model distribution with real RCI marked.

    Parameters
    ----------
    null_rcis : np.ndarray
        RCI values from null model shuffles
    real_rci : float
        RCI of real signal
    z_score : float
        Z-score of real RCI
    out_file : str, optional
        Output file path
    title : str
        Plot title
    figsize : tuple
        Figure size

    Examples
    --------
    >>> nulls = np.random.randn(100) * 0.1 + 0.2
    >>> # plot_null_distribution(nulls, real_rci=0.5, z_score=3.0)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Histogram of null distribution
    ax.hist(
        null_rcis, bins=30, alpha=0.7, color='gray',
        edgecolor='black', label='Null distribution'
    )

    # Mark real RCI
    ylim = ax.get_ylim()
    ax.axvline(
        real_rci, color='red', linewidth=2,
        label=f'Real RCI = {real_rci:.3f}\n(z = {z_score:.2f})'
    )

    # Mark mean and std of null
    null_mean = np.mean(null_rcis)
    null_std = np.std(null_rcis)
    ax.axvline(null_mean, color='blue', linestyle='--', linewidth=1.5, label=f'Null mean = {null_mean:.3f}')
    ax.axvspan(
        null_mean - null_std, null_mean + null_std,
        alpha=0.2, color='blue', label=f'Null std = {null_std:.3f}'
    )

    # Labels
    ax.set_xlabel('RCI Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def quick_visualize(
    signal: np.ndarray,
    arco_vector: np.ndarray,
    rational_list: List[Fraction],
    rci_global: float,
    rci_position: Optional[np.ndarray] = None,
    track_names: Optional[List[str]] = None,
    out_file: Optional[str] = None,
    title_prefix: str = "ARCO Analysis"
):
    """
    Create a comprehensive multi-panel visualization.

    Displays:
    1. Original signal
    2. Arcogram
    3. Polar arcogram
    4. RCI position profile (if available)

    Parameters
    ----------
    signal : np.ndarray
        Original signal
    arco_vector : np.ndarray
        ARCO fingerprint
    rational_list : List[Fraction]
        Rational arcs
    rci_global : float
        Global RCI
    rci_position : np.ndarray, optional
        Position-wise RCI
    track_names : List[str], optional
        Track names
    out_file : str, optional
        Output file path
    title_prefix : str
        Prefix for subplot titles

    Examples
    --------
    >>> sig = np.sin(2 * np.pi * 0.1 * np.arange(200))
    >>> from fractions import Fraction
    >>> from arc_rci.arco_core import arco_from_signal, generate_rational_arcs
    >>> arco, rci, rci_pos = arco_from_signal(
    ...     sig, sample_rate=1.0, tracks=[{'name': 'amp'}],
    ...     window_sizes=[50], max_q=11
    ... )
    >>> rats = generate_rational_arcs(11)
    >>> # quick_visualize(sig, arco, rats, rci, rci_pos)
    """
    # Determine layout
    if rci_position is not None:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Original signal
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(signal, linewidth=1, color='steelblue')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'{title_prefix}: Original Signal (RCI = {rci_global:.3f})', fontweight='bold')
    ax1.grid(alpha=0.3)

    # 2. Arcogram
    ax2 = fig.add_subplot(gs[1, 0])
    if arco_vector.ndim == 1:
        arco_matrix = arco_vector.reshape(1, -1)
    else:
        arco_matrix = arco_vector

    im = ax2.imshow(arco_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    arc_labels = [str(r) for r in rational_list[:arco_matrix.shape[1]]]
    ax2.set_xticks(np.arange(len(arc_labels))[::max(1, len(arc_labels)//10)])
    ax2.set_xticklabels(
        [arc_labels[i] for i in range(0, len(arc_labels), max(1, len(arc_labels)//10))],
        rotation=45, ha='right', fontsize=8
    )
    if track_names:
        ax2.set_yticks(np.arange(len(track_names)))
        ax2.set_yticklabels(track_names)
    ax2.set_xlabel('Rational Frequency')
    ax2.set_ylabel('Track')
    ax2.set_title('Arcogram', fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Power')

    # 3. Polar arcogram
    ax3 = fig.add_subplot(gs[1, 1], projection='polar')
    freqs = np.array([float(r) for r in rational_list[:len(arco_vector)]])
    theta = freqs * 2 * np.pi
    radius = arco_vector if arco_vector.ndim == 1 else arco_vector.mean(axis=0)

    denominators = np.array([r.denominator for r in rational_list[:len(radius)]])
    norm = mpl.colors.Normalize(vmin=denominators.min(), vmax=denominators.max())
    cmap_obj = plt.get_cmap('plasma')

    for i in range(len(theta)):
        color = cmap_obj(norm(denominators[i]))
        ax3.plot([theta[i], theta[i]], [0, radius[i]], color=color, linewidth=2, alpha=0.7)
        ax3.plot(theta[i], radius[i], 'o', color=color, markersize=6)

    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(1)
    ax3.set_title('Polar Arcogram', fontweight='bold', pad=20)

    # 4. RCI position (if available)
    if rci_position is not None:
        ax4 = fig.add_subplot(gs[2, :])
        x = np.arange(len(rci_position))
        ax4.plot(x, rci_position, linewidth=2, color='steelblue')
        ax4.fill_between(x, 0, rci_position, alpha=0.3, color='steelblue')
        ax4.axhline(rci_global, color='red', linestyle='--', label=f'Global RCI = {rci_global:.3f}')
        ax4.set_xlabel('Window Index')
        ax4.set_ylabel('RCI')
        ax4.set_title('RCI Position Profile', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
