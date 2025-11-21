"""
ARCO I/O - Input/output utilities for ARCO module

Provides functions to:
1. Load signals from numpy files
2. Load sequences from FASTA files
3. Save ARCO features to various formats
4. Adapt different input formats to ARCO pipeline
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
import csv
import json
from pathlib import Path


def load_signal_from_numpy(path: str) -> np.ndarray:
    """
    Load a 1D signal from a numpy file (.npy or .npz).

    Parameters
    ----------
    path : str
        Path to numpy file

    Returns
    -------
    np.ndarray
        1D signal array

    Examples
    --------
    >>> # signal = load_signal_from_numpy('data/signal.npy')
    """
    path = Path(path)

    if path.suffix == '.npy':
        data = np.load(path)
    elif path.suffix == '.npz':
        # For .npz, try to get first array
        npz = np.load(path)
        # Try common keys
        for key in ['signal', 'data', 'arr_0']:
            if key in npz:
                data = npz[key]
                break
        else:
            # Just get first array
            data = npz[list(npz.keys())[0]]
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    # Ensure 1D
    if data.ndim > 1:
        # Flatten or take first row/column
        if data.shape[0] == 1:
            data = data.flatten()
        elif data.shape[1] == 1:
            data = data.flatten()
        else:
            # Take first row
            data = data[0]

    return data.astype(np.float64)


def load_sequence_from_fasta(path: str) -> Tuple[str, str]:
    """
    Load a sequence from a FASTA file.

    Parameters
    ----------
    path : str
        Path to FASTA file

    Returns
    -------
    seq_id : str
        Sequence identifier
    sequence : str
        Sequence string

    Examples
    --------
    >>> # seq_id, seq = load_sequence_from_fasta('protein.fasta')
    """
    path = Path(path)

    with open(path, 'r') as f:
        lines = f.readlines()

    # Parse FASTA
    seq_id = ""
    sequence = ""

    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            seq_id = line[1:]
        else:
            sequence += line

    return seq_id, sequence


def sequence_to_numeric(sequence: str, encoding: str = 'simple') -> np.ndarray:
    """
    Convert a sequence string to numeric array.

    Parameters
    ----------
    sequence : str
        Sequence string (protein, DNA, etc.)
    encoding : str
        Encoding scheme:
        - 'simple': A=0, C=1, D=2, ... (alphabetical)
        - 'hydrophobicity': use hydrophobicity scale
        - 'charge': use charge values

    Returns
    -------
    np.ndarray
        Numeric array representation

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> numeric = sequence_to_numeric(seq, encoding='simple')
    >>> len(numeric) == len(seq)
    True
    """
    sequence = sequence.upper()

    if encoding == 'simple':
        # Alphabetical encoding
        unique_chars = sorted(set(sequence))
        char_to_num = {ch: i for i, ch in enumerate(unique_chars)}
        return np.array([char_to_num.get(ch, 0) for ch in sequence], dtype=np.float64)

    elif encoding == 'hydrophobicity':
        # Kyte-Doolittle hydrophobicity scale
        hydro = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        return np.array([hydro.get(ch, 0.0) for ch in sequence], dtype=np.float64)

    elif encoding == 'charge':
        # Charge at pH 7
        charge = {
            'D': -1, 'E': -1,  # Acidic
            'K': 1, 'R': 1, 'H': 0.5,  # Basic
        }
        return np.array([charge.get(ch, 0.0) for ch in sequence], dtype=np.float64)

    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def save_arco_npz(
    arco_vector: np.ndarray,
    out_path: str,
    rci_global: Optional[float] = None,
    rci_position: Optional[np.ndarray] = None,
    rational_list: Optional[list] = None,
    track_names: Optional[list] = None,
    meta: Optional[dict] = None
):
    """
    Save ARCO features to .npz file.

    Parameters
    ----------
    arco_vector : np.ndarray
        ARCO fingerprint vector
    out_path : str
        Output path (.npz)
    rci_global : float, optional
        Global RCI value
    rci_position : np.ndarray, optional
        Per-window RCI values
    rational_list : list, optional
        List of rational arcs (as strings)
    track_names : list, optional
        List of track names
    meta : dict, optional
        Additional metadata

    Examples
    --------
    >>> arco = np.random.rand(42)
    >>> # save_arco_npz(arco, 'output.npz', rci_global=0.5)
    """
    save_dict = {'arco_print': arco_vector}

    if rci_global is not None:
        save_dict['rci_global'] = rci_global

    if rci_position is not None:
        save_dict['rci_position'] = rci_position

    if rational_list is not None:
        # Convert Fractions to strings
        save_dict['rational_list'] = np.array([str(r) for r in rational_list])

    if track_names is not None:
        save_dict['track_names'] = np.array(track_names)

    if meta is not None:
        # Save metadata as JSON string
        save_dict['metadata'] = json.dumps(meta)

    np.savez(out_path, **save_dict)


def save_arco_csv(
    arco_vector: np.ndarray,
    out_path: str,
    rational_list: Optional[list] = None,
    meta: Optional[dict] = None
):
    """
    Save ARCO vector to CSV file.

    Parameters
    ----------
    arco_vector : np.ndarray
        ARCO fingerprint vector
    out_path : str
        Output CSV path
    rational_list : list, optional
        List of rational arcs for column headers
    meta : dict, optional
        Metadata to include as header

    Examples
    --------
    >>> arco = np.random.rand(10)
    >>> # save_arco_csv(arco, 'output.csv')
    """
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write metadata as comments
        if meta:
            for key, val in meta.items():
                writer.writerow([f'# {key}: {val}'])

        # Write header
        if rational_list:
            header = [str(r) for r in rational_list]
        else:
            header = [f'arc_{i}' for i in range(len(arco_vector))]
        writer.writerow(header)

        # Write data
        writer.writerow(arco_vector.tolist())


def load_arco_npz(path: str) -> Dict[str, Any]:
    """
    Load ARCO features from .npz file.

    Parameters
    ----------
    path : str
        Path to .npz file

    Returns
    -------
    dict
        Dictionary with keys: arco_print, rci_global, rci_position, etc.

    Examples
    --------
    >>> # data = load_arco_npz('output.npz')
    >>> # arco = data['arco_print']
    """
    npz = np.load(path, allow_pickle=True)
    result = {}

    for key in npz.keys():
        if key == 'metadata':
            # Parse JSON metadata
            result[key] = json.loads(str(npz[key]))
        else:
            result[key] = npz[key]

    return result


def load_xrd_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load XRD spectrum from CSV file (2-column: angle, intensity).

    This adapter allows ARCO to process XRD data from the EvolvMorph pipeline.

    Parameters
    ----------
    path : str
        Path to XRD CSV file

    Returns
    -------
    angles : np.ndarray
        2-theta angles
    intensities : np.ndarray
        Intensity values

    Examples
    --------
    >>> # angles, intensities = load_xrd_spectrum('xrd_pattern.csv')
    """
    data = np.loadtxt(path, delimiter=',')

    if data.ndim == 1:
        # Single column - assume just intensities
        return np.arange(len(data)), data
    else:
        # Two columns - angle, intensity
        return data[:, 0], data[:, 1]


def batch_load_signals(
    input_dir: str,
    pattern: str = '*.npy',
    max_files: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Batch load signals from a directory.

    Parameters
    ----------
    input_dir : str
        Input directory path
    pattern : str
        Glob pattern for files
    max_files : int, optional
        Maximum number of files to load

    Returns
    -------
    dict
        Dictionary mapping filenames to signal arrays

    Examples
    --------
    >>> # signals = batch_load_signals('data/', pattern='*.npy')
    """
    from pathlib import Path

    input_path = Path(input_dir)
    files = sorted(input_path.glob(pattern))

    if max_files:
        files = files[:max_files]

    signals = {}
    for file_path in files:
        try:
            signal = load_signal_from_numpy(str(file_path))
            signals[file_path.name] = signal
        except Exception as e:
            print(f"Warning: Failed to load {file_path.name}: {e}")

    return signals


def save_batch_arco(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    format: str = 'npz'
):
    """
    Save batch ARCO results.

    Parameters
    ----------
    results : dict
        Dictionary mapping sample names to result dictionaries
    output_dir : str
        Output directory
    format : str
        'npz' or 'csv'

    Examples
    --------
    >>> # results = {'sample1': {'arco_print': arco1, 'rci_global': 0.5}, ...}
    >>> # save_batch_arco(results, 'output/', format='npz')
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for sample_name, result in results.items():
        # Remove extension from sample name
        base_name = Path(sample_name).stem

        if format == 'npz':
            out_file = output_path / f"{base_name}_ARCO.npz"
            save_arco_npz(
                result['arco_print'],
                str(out_file),
                rci_global=result.get('rci_global'),
                rci_position=result.get('rci_position'),
                rational_list=result.get('rational_list'),
                track_names=result.get('track_names'),
                meta=result.get('meta')
            )
        elif format == 'csv':
            out_file = output_path / f"{base_name}_ARCO.csv"
            save_arco_csv(
                result['arco_print'],
                str(out_file),
                rational_list=result.get('rational_list'),
                meta=result.get('meta')
            )
        else:
            raise ValueError(f"Unknown format: {format}")
