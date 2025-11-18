"""
ARCO CLI - Command-line interface for batch ARCO processing

Provides command-line tools to:
1. Batch process signals from directory
2. Compute ARCO fingerprints and RCI
3. Generate visualizations
4. Run null model validation
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import json

# Import ARCO modules
try:
    from .arco_core import arco_from_signal, generate_rational_arcs, null_model_zscore
    from .arco_io import (
        batch_load_signals, save_batch_arco, load_signal_from_numpy,
        save_arco_npz
    )
    from .arco_vis import plot_arcogram, plot_polar_arcogram, quick_visualize
except ImportError:
    # Handle case when run as script
    from arco_core import arco_from_signal, generate_rational_arcs, null_model_zscore
    from arco_io import (
        batch_load_signals, save_batch_arco, load_signal_from_numpy,
        save_arco_npz
    )
    from arco_vis import plot_arcogram, plot_polar_arcogram, quick_visualize


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='ARCO: Rational-frequency spectral fingerprinting and RCI computation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all .npy files in a directory
  python -m arc_rci.arco_cli --input signals/ --pattern "*.npy" --out feats/ --fs 500

  # Single file with custom parameters
  python -m arc_rci.arco_cli --input signal.npy --out output/ --fs 1.0 --max-q 30 --window-sizes 50 100

  # With null model and visualization
  python -m arc_rci.arco_cli --input signal.npy --out output/ --null-model --visualize
        """
    )

    # Input/output
    parser.add_argument(
        '--input', '-i', required=True,
        help='Input file or directory'
    )
    parser.add_argument(
        '--pattern', '-p', default='*.npy',
        help='File pattern for batch processing (default: *.npy)'
    )
    parser.add_argument(
        '--out', '-o', required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--format', '-f', default='npz', choices=['npz', 'csv'],
        help='Output format (default: npz)'
    )

    # Signal parameters
    parser.add_argument(
        '--fs', '--sample-rate', type=float, default=1.0,
        help='Sampling rate in Hz (default: 1.0 for sequences)'
    )
    parser.add_argument(
        '--tracks', nargs='+', default=['amp'],
        choices=['amp', 'deriv', 'deriv2', 'envelope', 'diff'],
        help='Tracks to compute (default: amp)'
    )

    # ARCO parameters
    parser.add_argument(
        '--max-q', type=int, default=11,
        help='Maximum denominator for rational arcs (default: 11)'
    )
    parser.add_argument(
        '--window-sizes', nargs='+', type=int, default=[31, 63],
        help='Window sizes in samples (default: 31 63)'
    )
    parser.add_argument(
        '--step-fraction', type=float, default=0.25,
        help='Step size as fraction of window (default: 0.25 = 75%% overlap)'
    )
    parser.add_argument(
        '--delta-factor', type=float, default=1.0,
        help='Bandwidth scaling factor (default: 1.0)'
    )
    parser.add_argument(
        '--integration-scheme', default='gaussian',
        choices=['gaussian', 'hard', 'triangular'],
        help='Integration scheme (default: gaussian)'
    )

    # Null model
    parser.add_argument(
        '--null-model', action='store_true',
        help='Compute null model z-scores'
    )
    parser.add_argument(
        '--n-shuffles', type=int, default=50,
        help='Number of shuffles for null model (default: 50)'
    )
    parser.add_argument(
        '--null-type', default='composition',
        choices=['composition', 'markov1'],
        help='Null model type (default: composition)'
    )

    # Visualization
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--viz-type', nargs='+', default=['arcogram', 'polar'],
        choices=['arcogram', 'polar', 'quick'],
        help='Visualization types (default: arcogram polar)'
    )

    # Other
    parser.add_argument(
        '--max-files', type=int,
        help='Maximum number of files to process (for testing)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--seed', type=int,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


def process_single_signal(
    signal: np.ndarray,
    signal_name: str,
    args,
    rational_list: List
) -> Dict[str, Any]:
    """Process a single signal and return results."""

    if args.verbose:
        print(f"  Processing signal: {signal_name} (length={len(signal)})")

    # Convert tracks to dict format
    tracks = [{'name': t} for t in args.tracks]

    # Compute ARCO
    try:
        arco_print, rci_global, rci_position = arco_from_signal(
            signal,
            sample_rate=args.fs,
            tracks=tracks,
            window_sizes=args.window_sizes,
            step_fraction=args.step_fraction,
            max_q=args.max_q,
            delta_factor=args.delta_factor,
            integration_scheme=args.integration_scheme
        )

        if args.verbose:
            print(f"    ARCO shape: {arco_print.shape}, RCI: {rci_global:.4f}")

    except Exception as e:
        print(f"  Error computing ARCO for {signal_name}: {e}")
        return None

    # Result dictionary
    result = {
        'arco_print': arco_print,
        'rci_global': rci_global,
        'rci_position': rci_position,
        'rational_list': rational_list,
        'track_names': args.tracks,
        'meta': {
            'sample_rate': args.fs,
            'max_q': args.max_q,
            'window_sizes': args.window_sizes,
            'step_fraction': args.step_fraction,
            'delta_factor': args.delta_factor,
            'signal_length': len(signal)
        }
    }

    # Null model
    if args.null_model:
        if args.verbose:
            print(f"    Running null model ({args.n_shuffles} shuffles)...")

        try:
            z_score, null_rcis, real_rci, p_value = null_model_zscore(
                signal,
                n_shuffles=args.n_shuffles,
                null_type=args.null_type,
                seed=args.seed,
                sample_rate=args.fs,
                tracks=tracks,
                window_sizes=args.window_sizes,
                step_fraction=args.step_fraction,
                max_q=args.max_q,
                delta_factor=args.delta_factor,
                integration_scheme=args.integration_scheme
            )

            result['null_zscore'] = z_score
            result['null_pvalue'] = p_value
            result['null_rcis'] = null_rcis

            if args.verbose:
                print(f"    Null model: z={z_score:.2f}, p={p_value:.4f}")

        except Exception as e:
            print(f"  Warning: Null model failed for {signal_name}: {e}")

    return result


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)

    # Create output directory
    output_path = Path(args.out)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate rational arcs
    rational_list = generate_rational_arcs(args.max_q)
    if args.verbose:
        print(f"Generated {len(rational_list)} rational arcs (max_q={args.max_q})")

    # Determine if input is file or directory
    input_path = Path(args.input)

    if input_path.is_file():
        # Process single file
        if args.verbose:
            print(f"Processing single file: {input_path}")

        signal = load_signal_from_numpy(str(input_path))
        signal_name = input_path.name

        result = process_single_signal(signal, signal_name, args, rational_list)

        if result:
            # Save result
            base_name = input_path.stem
            if args.format == 'npz':
                out_file = output_path / f"{base_name}_ARCO.npz"
                save_arco_npz(
                    result['arco_print'],
                    str(out_file),
                    rci_global=result['rci_global'],
                    rci_position=result['rci_position'],
                    rational_list=result['rational_list'],
                    track_names=result['track_names'],
                    meta=result['meta']
                )
                if 'null_zscore' in result:
                    # Save null model results separately
                    null_file = output_path / f"{base_name}_null.npz"
                    np.savez(
                        null_file,
                        z_score=result['null_zscore'],
                        p_value=result['null_pvalue'],
                        null_rcis=result['null_rcis']
                    )

            # Visualization
            if args.visualize:
                if args.verbose:
                    print(f"  Generating visualizations...")

                if 'arcogram' in args.viz_type:
                    plot_arcogram(
                        result['arco_print'].reshape(1, -1) if result['arco_print'].ndim == 1 else result['arco_print'],
                        result['rational_list'],
                        track_names=result['track_names'],
                        out_file=str(output_path / f"{base_name}_arcogram.png"),
                        title=f"Arcogram: {signal_name}"
                    )

                if 'polar' in args.viz_type:
                    arco_vec = result['arco_print'] if result['arco_print'].ndim == 1 else result['arco_print'].mean(axis=0)
                    plot_polar_arcogram(
                        arco_vec,
                        result['rational_list'],
                        out_file=str(output_path / f"{base_name}_polar.png"),
                        title=f"Polar Arcogram: {signal_name}"
                    )

                if 'quick' in args.viz_type:
                    quick_visualize(
                        signal,
                        result['arco_print'],
                        result['rational_list'],
                        result['rci_global'],
                        rci_position=result['rci_position'],
                        track_names=result['track_names'],
                        out_file=str(output_path / f"{base_name}_quick.png"),
                        title_prefix=signal_name
                    )

            print(f"✓ Results saved to {output_path}")
            print(f"  RCI: {result['rci_global']:.4f}")
            if 'null_zscore' in result:
                print(f"  Null model z-score: {result['null_zscore']:.2f} (p={result['null_pvalue']:.4f})")

    elif input_path.is_dir():
        # Batch processing
        if args.verbose:
            print(f"Batch processing directory: {input_path}")
            print(f"  Pattern: {args.pattern}")

        signals = batch_load_signals(
            str(input_path),
            pattern=args.pattern,
            max_files=args.max_files
        )

        if not signals:
            print(f"Error: No signals found in {input_path} matching pattern {args.pattern}")
            sys.exit(1)

        print(f"Loaded {len(signals)} signals")

        # Process each signal
        results = {}
        for i, (signal_name, signal) in enumerate(signals.items(), 1):
            print(f"[{i}/{len(signals)}] {signal_name}")

            result = process_single_signal(signal, signal_name, args, rational_list)

            if result:
                results[signal_name] = result

        # Save batch results
        if results:
            save_batch_arco(results, str(output_path), format=args.format)

            # Save summary
            summary = {
                'n_signals': len(results),
                'rci_mean': float(np.mean([r['rci_global'] for r in results.values()])),
                'rci_std': float(np.std([r['rci_global'] for r in results.values()])),
                'rci_min': float(np.min([r['rci_global'] for r in results.values()])),
                'rci_max': float(np.max([r['rci_global'] for r in results.values()])),
                'parameters': {
                    'max_q': args.max_q,
                    'window_sizes': args.window_sizes,
                    'step_fraction': args.step_fraction,
                    'sample_rate': args.fs,
                    'tracks': args.tracks
                }
            }

            if args.null_model:
                summary['null_z_mean'] = float(np.mean([r.get('null_zscore', 0) for r in results.values()]))

            summary_file = output_path / 'summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"\n✓ Batch processing complete!")
            print(f"  Processed: {len(results)} signals")
            print(f"  RCI mean: {summary['rci_mean']:.4f} ± {summary['rci_std']:.4f}")
            print(f"  Results saved to: {output_path}")

    else:
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
