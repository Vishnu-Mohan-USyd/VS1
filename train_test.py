#!/usr/bin/env python3
"""
Training test: Run STDP training and show OSI improvement.
"""

import numpy as np
import sys
sys.modules['cupy'] = None

from network import VisualCortexNetwork
from stimuli import DriftingGratingGenerator
from config import GRATING_ORIENTATIONS, SIM_DT

# Very short durations for quick testing
PRESENTATION_DURATION = 50   # ms
INTER_STIMULUS_INTERVAL = 10  # ms

def measure_osi(net, grating, n_trials=1, duration_ms=30):
    """Measure OSI for first hypercolumn."""
    n_ens = net.n_orientations
    neurons_per_ens = net.neurons_per_ensemble

    responses = np.zeros((n_ens, len(GRATING_ORIENTATIONS)))

    for ori_idx, ori in enumerate(GRATING_ORIENTATIONS):
        for trial in range(n_trials):
            # Reset neuron states
            net.v1_exc.v[:] = net.v1_exc.v_rest
            net.v1_exc.u[:] = net.v1_exc.b * net.v1_exc.v_rest
            net.v1_inh.v[:] = net.v1_inh.v_rest
            net.v1_inh.u[:] = net.v1_inh.b * net.v1_inh.v_rest
            net.v1_exc.g_exc[:] = 0
            net.v1_exc.g_inh[:] = 0
            net.v1_inh.g_exc[:] = 0
            net.v1_inh.g_inh[:] = 0

            n_ts = int(duration_ms / SIM_DT)
            for t in range(n_ts):
                time_ms = t * SIM_DT
                on_spikes, off_spikes = grating.generate_spikes(time_ms, ori)
                net.step(time_ms, on_spikes, off_spikes, learning=False)

                exc_spikes = np.array(net.v1_exc.spikes, dtype=bool)
                for ens in range(n_ens):
                    start_idx = ens * neurons_per_ens
                    end_idx = start_idx + neurons_per_ens
                    responses[ens, ori_idx] += np.sum(exc_spikes[start_idx:end_idx])

    # Average over trials
    responses /= n_trials

    # Calculate OSI for each ensemble
    osi_values = []
    preferred_orientations = []
    for ens in range(n_ens):
        r = responses[ens, :]
        if np.sum(r) > 0:
            theta = np.deg2rad(2 * GRATING_ORIENTATIONS)
            complex_sum = np.sum(r * np.exp(1j * theta))
            osi = np.abs(complex_sum) / np.sum(r)
            preferred = np.rad2deg(np.angle(complex_sum) / 2) % 180
        else:
            osi = 0
            preferred = 0
        osi_values.append(osi)
        preferred_orientations.append(preferred)

    return np.array(osi_values), np.array(preferred_orientations), responses

def train_one_epoch(net, grating, epoch_num, presentations_per_ori=5):
    """Train for one epoch (all orientations)."""
    print(f"\n--- Epoch {epoch_num} ---")

    total_exc_spikes = 0
    total_inh_spikes = 0
    n_timesteps = 0

    for ori in GRATING_ORIENTATIONS:
        for pres in range(presentations_per_ori):
            # Presentation phase (with learning)
            n_ts = int(PRESENTATION_DURATION / SIM_DT)
            for t in range(n_ts):
                time_ms = t * SIM_DT
                on_spikes, off_spikes = grating.generate_spikes(time_ms, ori)
                result = net.step(time_ms, on_spikes, off_spikes, learning=True)
                total_exc_spikes += result['v1_exc']
                total_inh_spikes += result['v1_inh']
                n_timesteps += 1

            # Inter-stimulus interval (no stimulus, no learning)
            n_ts_isi = int(INTER_STIMULUS_INTERVAL / SIM_DT)
            for t in range(n_ts_isi):
                time_ms = t * SIM_DT
                blank_on = np.zeros_like(on_spikes)
                blank_off = np.zeros_like(off_spikes)
                net.step(time_ms, blank_on, blank_off, learning=False)

    # Average firing rate during epoch
    duration_s = (n_timesteps * SIM_DT) / 1000.0
    exc_rate = total_exc_spikes / (net.v1_exc.n_neurons * duration_s)
    inh_rate = total_inh_spikes / (net.v1_inh.n_neurons * duration_s)

    print(f"  Avg firing rate: Exc={exc_rate:.1f} Hz, Inh={inh_rate:.1f} Hz")

    # Get weight statistics
    weights = net.get_lgn_v1_weights()
    nonzero = weights[weights > 0]
    print(f"  Weights: mean={np.mean(nonzero):.4f}, std={np.std(nonzero):.4f}, max={np.max(nonzero):.4f}")

def main():
    print("=" * 60)
    print("STDP Training Test")
    print("=" * 60)

    np.random.seed(42)

    # Create network and stimulus
    net = VisualCortexNetwork()
    grating = DriftingGratingGenerator()

    # Measure initial OSI
    print("\n" + "=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)

    osi_before, pref_before, resp_before = measure_osi(net, grating, n_trials=1, duration_ms=50)
    print("\nOSI values (Hypercolumn 0):")
    for ens in range(len(osi_before)):
        pref = GRATING_ORIENTATIONS[ens]
        print(f"  Ensemble {ens} (assigned {pref:.0f}°): OSI={osi_before[ens]:.3f}, pref={pref_before[ens]:.1f}°")
    print(f"\nMean OSI: {np.mean(osi_before):.3f}")

    # Training
    print("\n" + "=" * 60)
    print("TRAINING (3 epochs)")
    print("=" * 60)

    n_epochs = 5
    presentations_per_ori = 3

    for epoch in range(1, n_epochs + 1):
        train_one_epoch(net, grating, epoch, presentations_per_ori)

        # Measure OSI after each epoch
        osi_current, _, _ = measure_osi(net, grating, n_trials=1, duration_ms=50)
        print(f"  OSI after epoch {epoch}: mean={np.mean(osi_current):.3f}, max={np.max(osi_current):.3f}")

    # Measure final OSI
    print("\n" + "=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)

    osi_after, pref_after, resp_after = measure_osi(net, grating, n_trials=2, duration_ms=50)
    print("\nOSI values (Hypercolumn 0):")
    for ens in range(len(osi_after)):
        assigned = GRATING_ORIENTATIONS[ens]
        print(f"  Ensemble {ens} (assigned {assigned:.0f}°): OSI={osi_after[ens]:.3f}, pref={pref_after[ens]:.1f}°")
    print(f"\nMean OSI: {np.mean(osi_after):.3f}")

    # Improvement
    print("\n" + "=" * 60)
    print("IMPROVEMENT")
    print("=" * 60)
    print(f"Mean OSI before: {np.mean(osi_before):.3f}")
    print(f"Mean OSI after:  {np.mean(osi_after):.3f}")
    improvement = (np.mean(osi_after) - np.mean(osi_before)) / np.mean(osi_before) * 100
    print(f"Improvement: {improvement:+.1f}%")

    # Print tuning curves
    print("\n" + "=" * 60)
    print("TUNING CURVES (After Training)")
    print("=" * 60)
    print("\nEnsemble | " + " | ".join([f"{o:5.1f}°" for o in GRATING_ORIENTATIONS]))
    print("-" * 80)
    for ens in range(len(osi_after)):
        pref = GRATING_ORIENTATIONS[ens]
        vals = " | ".join([f"{resp_after[ens, i]:6.0f}" for i in range(len(GRATING_ORIENTATIONS))])
        print(f"   {ens} ({pref:5.1f}°) | {vals}")

    print("\n" + "=" * 60)
    print("Training complete")

if __name__ == "__main__":
    main()
