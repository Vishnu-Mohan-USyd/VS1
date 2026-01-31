#!/usr/bin/env python3
"""
Main simulation script for V1 Orientation Selectivity Development.

This script implements a spiking neural network model of the early visual
pathway (RGC -> LGN -> V1 Layer 4) that develops orientation selectivity
through STDP learning on drifting grating stimuli.

Usage:
    python main.py                    # Run with default parameters
    python main.py --epochs 5         # Run 5 training epochs
    python main.py --test-only        # Only run testing (load existing weights)

Author: Based on Srinivasa & Jiang (2013) and classic feedforward models
"""

import argparse
import os
import sys
import time
import pickle
from datetime import datetime

import numpy as np

# Ensure output directory exists
os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Check for GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled (CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    print("Running on CPU (install CuPy for GPU acceleration)")

from config import (
    N_ORIENTATIONS, GRATING_ORIENTATIONS, RETINA_SIZE,
    N_TRAINING_EPOCHS, N_PRESENTATIONS_PER_ORI,
    PRESENTATION_DURATION, INTER_STIMULUS_INTERVAL,
    SIM_DT, N_HYPERCOLUMNS
)
from neurons import to_numpy, to_gpu
from network import VisualCortexNetwork
from stimuli import DriftingGratingGenerator, create_blank_stimulus
from visualization import (
    compute_osi, compute_preferred_orientation,
    plot_weight_heatmap, plot_ensemble_weight_summary,
    plot_osi_distribution, plot_tuning_curves,
    plot_orientation_map, plot_training_progress,
    create_training_summary, OUTPUT_DIR
)


def measure_responses(network, stimulus_gen, orientations, duration=500, n_trials=3):
    """
    Measure V1 responses to each orientation.

    Args:
        network: VisualCortexNetwork instance
        stimulus_gen: DriftingGratingGenerator instance
        orientations: Array of orientations to test
        duration: Stimulus duration per trial (ms)
        n_trials: Number of trials per orientation

    Returns:
        responses: Array of shape (n_ensembles, n_orientations)
    """
    n_ensembles = network.n_hypercolumns * network.n_orientations
    responses = np.zeros((n_ensembles, len(orientations)))

    print(f"Measuring responses to {len(orientations)} orientations...")

    for ori_idx, orientation in enumerate(orientations):
        trial_responses = []

        for trial in range(n_trials):
            # Reset network state but keep weights
            network.reset()

            # Initialize spike counts
            ensemble_spikes = np.zeros(n_ensembles)

            # Present stimulus
            for t in np.arange(0, duration, SIM_DT):
                # Generate spikes
                on_spikes, off_spikes = stimulus_gen.generate_spikes(orientation, t)

                # Run network (no learning during testing)
                network.step(t, on_spikes, off_spikes, learning=False)

                # Count spikes per ensemble
                v1_spikes = to_numpy(network.v1_exc.spikes)
                for hc in range(network.n_hypercolumns):
                    for ori in range(network.n_orientations):
                        ens_idx = hc * network.n_orientations + ori
                        neuron_indices = network.exc_idx[hc, ori, :].flatten()
                        ensemble_spikes[ens_idx] += np.sum(v1_spikes[neuron_indices])

            trial_responses.append(ensemble_spikes)

        # Average across trials
        responses[:, ori_idx] = np.mean(trial_responses, axis=0)

    return responses


def train_epoch(network, stimulus_gen, orientations, n_presentations,
                presentation_duration, isi, epoch_num, history):
    """
    Run one training epoch through all orientations.

    Args:
        network: VisualCortexNetwork instance
        stimulus_gen: DriftingGratingGenerator instance
        orientations: Array of orientations
        n_presentations: Presentations per orientation
        presentation_duration: Duration of each presentation (ms)
        isi: Inter-stimulus interval (ms)
        epoch_num: Current epoch number
        history: Dictionary to store training history

    Returns:
        Updated history dictionary
    """
    # Create randomized presentation order
    presentation_order = []
    for _ in range(n_presentations):
        order = np.random.permutation(len(orientations))
        presentation_order.extend(orientations[order])

    total_presentations = len(presentation_order)
    trial_duration = presentation_duration + isi

    print(f"\nEpoch {epoch_num}: {total_presentations} presentations")
    print("-" * 50)

    global_time = 0
    start_time = time.time()

    for pres_idx, orientation in enumerate(presentation_order):
        # Progress update
        if pres_idx % 10 == 0:
            elapsed = time.time() - start_time
            progress = (pres_idx + 1) / total_presentations
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  Presentation {pres_idx + 1}/{total_presentations} "
                  f"(Ori: {orientation:.0f}°) - ETA: {eta:.0f}s", end='\r')

        # Present stimulus
        for t_local in np.arange(0, presentation_duration, SIM_DT):
            t = global_time + t_local
            on_spikes, off_spikes = stimulus_gen.generate_spikes(orientation, t_local)
            spike_counts = network.step(t, on_spikes, off_spikes, learning=True)

        # ISI (blank screen)
        for t_local in np.arange(0, isi, SIM_DT):
            t = global_time + presentation_duration + t_local
            on_spikes, off_spikes = create_blank_stimulus(RETINA_SIZE)
            spike_counts = network.step(t, on_spikes, off_spikes, learning=True)

        global_time += trial_duration

        # Record history periodically
        if pres_idx % 5 == 0:
            weights = network.get_lgn_v1_weights()
            active_weights = weights[weights > 0]

            history['time'].append(global_time)
            history['v1_exc_rates'].append(spike_counts['v1_exc'] * 1000 / SIM_DT)
            history['v1_inh_rates'].append(spike_counts['v1_inh'] * 1000 / SIM_DT)
            history['lgn_rates'].append((spike_counts['lgn_on'] + spike_counts['lgn_off']) * 500 / SIM_DT)

            if len(active_weights) > 0:
                history['mean_weight'].append(np.mean(active_weights))
                history['max_weight'].append(np.max(active_weights))
                history['min_weight'].append(np.min(active_weights[active_weights > 0.01]))
            else:
                history['mean_weight'].append(0)
                history['max_weight'].append(0)
                history['min_weight'].append(0)

            stats = network.lgn_v1_stdp.get_stats()
            history['total_pot'].append(stats['total_pot'])
            history['total_dep'].append(stats['total_dep'])

    print(f"\n  Epoch {epoch_num} completed in {time.time() - start_time:.1f}s")

    return history


def save_checkpoint(network, epoch, filepath):
    """Save network weights to file."""
    checkpoint = {
        'epoch': epoch,
        'lgn_v1_weights': network.get_lgn_v1_weights(),
        'w_inh_exc': to_numpy(network.w_inh_exc),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(network, filepath):
    """Load network weights from file."""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)

    network.lgn_v1_stdp.weights = to_gpu(checkpoint['lgn_v1_weights'])
    network.w_inh_exc = to_gpu(checkpoint['w_inh_exc'])
    network.inh_stdp.weights = network.w_inh_exc

    print(f"Checkpoint loaded: {filepath} (epoch {checkpoint['epoch']})")
    return checkpoint['epoch']


def main():
    """Main training and evaluation loop."""
    parser = argparse.ArgumentParser(description='V1 Orientation Selectivity Development')
    parser.add_argument('--epochs', type=int, default=N_TRAINING_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--presentations', type=int, default=N_PRESENTATIONS_PER_ORI,
                       help='Presentations per orientation per epoch')
    parser.add_argument('--duration', type=int, default=PRESENTATION_DURATION,
                       help='Presentation duration (ms)')
    parser.add_argument('--isi', type=int, default=INTER_STIMULUS_INTERVAL,
                       help='Inter-stimulus interval (ms)')
    parser.add_argument('--test-only', action='store_true',
                       help='Skip training, only run testing')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to checkpoint to load')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    if GPU_AVAILABLE:
        cp.random.seed(args.seed)

    print("=" * 60)
    print("V1 Orientation Selectivity Development")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Presentations per orientation: {args.presentations}")
    print(f"Presentation duration: {args.duration} ms")
    print(f"ISI: {args.isi} ms")
    print(f"Orientations: {GRATING_ORIENTATIONS}")
    print("=" * 60)

    # Initialize network
    print("\nInitializing network...")
    network = VisualCortexNetwork()

    # Initialize stimulus generator
    stimulus_gen = DriftingGratingGenerator(size=RETINA_SIZE)

    # Load checkpoint if specified
    start_epoch = 0
    if args.load:
        start_epoch = load_checkpoint(network, args.load)

    # Training history
    history = {
        'time': [],
        'lgn_rates': [],
        'v1_exc_rates': [],
        'v1_inh_rates': [],
        'mean_weight': [],
        'max_weight': [],
        'min_weight': [],
        'total_pot': [],
        'total_dep': [],
        'mean_osi': [],
        'max_osi': [],
        'weight_hist': [],
        'weight_bins': [],
        'weight_hist_times': [],
    }

    # Pre-training evaluation
    print("\n" + "=" * 60)
    print("PRE-TRAINING EVALUATION")
    print("=" * 60)

    responses_pre = measure_responses(
        network, stimulus_gen, GRATING_ORIENTATIONS,
        duration=args.duration, n_trials=3
    )
    summary_pre = create_training_summary(
        network, responses_pre, GRATING_ORIENTATIONS,
        epoch=0, save_dir=OUTPUT_DIR
    )

    if not args.test_only:
        # Training loop
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            # Train one epoch
            history = train_epoch(
                network, stimulus_gen, GRATING_ORIENTATIONS,
                args.presentations, args.duration, args.isi,
                epoch, history
            )

            # Measure OSI and update history
            print(f"\nEvaluating epoch {epoch}...")
            responses = measure_responses(
                network, stimulus_gen, GRATING_ORIENTATIONS,
                duration=args.duration, n_trials=2
            )

            osi_values = np.array([
                compute_osi(responses[i], GRATING_ORIENTATIONS)
                for i in range(len(responses))
            ])
            history['mean_osi'].append(np.mean(osi_values))
            history['max_osi'].append(np.max(osi_values))

            # Save weight histogram
            weights = network.get_lgn_v1_weights()
            active_weights = weights[weights > 0]
            if len(active_weights) > 0:
                hist, bins = np.histogram(active_weights, bins=50)
                history['weight_hist'].append(hist)
                history['weight_bins'].append(bins)
                history['weight_hist_times'].append(history['time'][-1] if history['time'] else 0)

            # Create summary plots
            summary = create_training_summary(
                network, responses, GRATING_ORIENTATIONS,
                epoch=epoch, save_dir=OUTPUT_DIR
            )

            # Save checkpoint
            save_checkpoint(
                network, epoch,
                f"checkpoints/epoch_{epoch}.pkl"
            )

        # Plot training progress
        plot_training_progress(history, save_path=f"{OUTPUT_DIR}/training_progress.png")

    # Post-training evaluation
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    responses_post = measure_responses(
        network, stimulus_gen, GRATING_ORIENTATIONS,
        duration=args.duration, n_trials=5
    )
    summary_post = create_training_summary(
        network, responses_post, GRATING_ORIENTATIONS,
        epoch=args.epochs if not args.test_only else 'final',
        save_dir=OUTPUT_DIR
    )

    # Compare before and after
    print("\n" + "=" * 60)
    print("COMPARISON: Before vs After Training")
    print("=" * 60)

    osi_pre = np.array([compute_osi(responses_pre[i], GRATING_ORIENTATIONS)
                       for i in range(len(responses_pre))])
    osi_post = np.array([compute_osi(responses_post[i], GRATING_ORIENTATIONS)
                        for i in range(len(responses_post))])

    print(f"Mean OSI:   {np.mean(osi_pre):.4f} -> {np.mean(osi_post):.4f} "
          f"(change: {np.mean(osi_post) - np.mean(osi_pre):+.4f})")
    print(f"Median OSI: {np.median(osi_pre):.4f} -> {np.median(osi_post):.4f}")
    print(f"Max OSI:    {np.max(osi_pre):.4f} -> {np.max(osi_post):.4f}")
    print(f"OSI > 0.3:  {np.mean(osi_pre > 0.3):.1%} -> {np.mean(osi_post > 0.3):.1%}")
    print(f"OSI > 0.5:  {np.mean(osi_pre > 0.5):.1%} -> {np.mean(osi_post > 0.5):.1%}")

    # Create comparison figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # OSI distributions
    ax = axes[0, 0]
    ax.hist(osi_pre, bins=30, alpha=0.5, label='Before', density=True)
    ax.hist(osi_post, bins=30, alpha=0.5, label='After', density=True)
    ax.axvline(np.mean(osi_pre), color='blue', linestyle='--')
    ax.axvline(np.mean(osi_post), color='orange', linestyle='--')
    ax.set_xlabel("OSI")
    ax.set_ylabel("Density")
    ax.set_title("OSI Distribution")
    ax.legend()

    # Weight distributions
    ax = axes[0, 1]
    weights = network.get_lgn_v1_weights()
    active_weights = weights[weights > 0]
    ax.hist(active_weights, bins=50, edgecolor='black')
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title("Final Weight Distribution")

    # Tuning curve comparison
    ax = axes[1, 0]
    # Pick an ensemble with high OSI improvement
    osi_improvement = osi_post - osi_pre
    best_improvement_idx = np.argmax(osi_improvement)
    ax.plot(GRATING_ORIENTATIONS, responses_pre[best_improvement_idx], 'o-',
           label=f'Before (OSI={osi_pre[best_improvement_idx]:.2f})')
    ax.plot(GRATING_ORIENTATIONS, responses_post[best_improvement_idx], 's-',
           label=f'After (OSI={osi_post[best_improvement_idx]:.2f})')
    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel("Response (spikes)")
    ax.set_title(f"Tuning Curve (Ensemble {best_improvement_idx})")
    ax.legend()

    # Weight heatmap for best ensemble's hypercolumn
    ax = axes[1, 1]
    best_hc = best_improvement_idx // N_ORIENTATIONS
    ensemble_weights = network.get_all_ensemble_weights()
    ax.bar(range(N_ORIENTATIONS), ensemble_weights[best_hc])
    ax.set_xlabel("Orientation Index")
    ax.set_ylabel("Total Weight")
    ax.set_title(f"Weights for Hypercolumn {best_hc}")
    ax.set_xticks(range(N_ORIENTATIONS))
    ax.set_xticklabels([f"{i * 180 // N_ORIENTATIONS}°" for i in range(N_ORIENTATIONS)])

    plt.suptitle("Training Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
