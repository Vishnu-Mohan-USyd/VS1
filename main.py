#!/usr/bin/env python3
"""
Main entry point for the Spiking Neural Network Visual Pathway Simulation.

This simulation models the visual pathway from Retinal Ganglion Cells (RGC)
through the Lateral Geniculate Nucleus (LGN) to Layer 4 of Primary Visual
Cortex (V1), demonstrating how orientation selectivity can emerge through
STDP learning and lateral competition.

Architecture:
- RGC: ON and OFF center-surround cells
- LGN: Retinotopic relay of ON/OFF signals
- V1: Multiple orientation ensembles per hypercolumn (retinotopic patch)

Learning:
- STDP (Spike-Timing Dependent Plasticity) for LGN -> V1 connections
- Lateral inhibition between V1 ensembles for competition
- Homeostatic plasticity for activity normalization

Usage:
    python main.py                  # Run full training
    python main.py --test           # Run quick test
    python main.py --demo           # Generate stimulus demos
    python main.py --epochs 20      # Train for specific epochs
    python main.py --cpu            # Force CPU (no GPU)

Author: Claude
"""

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Spiking Neural Network for Orientation Selectivity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run full training (default)
  python main.py --test             Quick test of all components
  python main.py --demo             Generate stimulus visualizations
  python main.py --epochs 50        Train for 50 epochs
  python main.py --output results   Save output to 'results' folder
  python main.py --cpu              Force CPU computation

Output:
  The simulation generates visualizations in the output directory:
  - weights_heatmap_epochXXX.png    Weight matrices for each ensemble
  - tuning_curves_epochXXX.png      Orientation tuning curves
  - orientation_map_epochXXX.png    Orientation preference map
  - selectivity_evolution.png       OSI over training
  - training_summary.png            Comprehensive summary

Notes:
  - GPU acceleration requires CuPy with CUDA support
  - Training time depends on epochs and visual field size
  - Default configuration trains 8 orientation ensembles
        """
    )

    parser.add_argument('--test', action='store_true',
                       help='Run quick test of all components')

    parser.add_argument('--demo', action='store_true',
                       help='Generate stimulus visualizations only')

    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default from config)')

    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory (default: output)')

    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU computation (disable GPU)')

    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Verbose output (default: True)')

    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode (minimal output)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle CPU flag by modifying config
    if args.cpu:
        import config
        config.USE_GPU = False
        print("Forcing CPU mode (GPU disabled)")

    # Handle verbosity
    verbose = not args.quiet

    # Import simulation (after config modification)
    from simulation import VisualPathwaySimulation, QuickTest
    from config import N_TRAINING_EPOCHS

    if args.test:
        # Run quick test
        QuickTest.run(output_dir=args.output)

    elif args.demo:
        # Generate stimulus demos only
        sim = VisualPathwaySimulation(output_dir=args.output, verbose=verbose)
        for ori in [0, 45, 90, 135]:
            sim.demo_stimulus(ori)
        print(f"\nStimulus demos saved to {args.output}/")

    else:
        # Full training
        n_epochs = args.epochs if args.epochs else N_TRAINING_EPOCHS

        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Output will be saved to: {args.output}/")

        sim = VisualPathwaySimulation(output_dir=args.output, verbose=verbose)

        # Generate stimulus demos
        sim.demo_stimulus(45)
        sim.demo_stimulus(90)

        # Run training
        sim.train(n_epochs=n_epochs)

        print("\nTraining complete!")
        print(f"Check {args.output}/ for visualizations and results.")


if __name__ == "__main__":
    main()
