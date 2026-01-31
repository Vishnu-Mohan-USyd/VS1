#!/usr/bin/env python3
"""Quick diagnostic test for the V1 network."""

import numpy as np
import sys

# Force CPU for faster startup
sys.modules['cupy'] = None

from network import VisualCortexNetwork
from stimuli import DriftingGratingGenerator
from config import GRATING_ORIENTATIONS, SIM_DT

def main():
    print("Quick Network Diagnostic Test")
    print("=" * 60)

    # Create network
    net = VisualCortexNetwork()

    # Create stimulus
    grating = DriftingGratingGenerator()

    # Test with one orientation (0 degrees)
    orientation = 0.0

    print(f"\nTesting with {orientation} degree grating for 100ms...")

    # Track spikes per ensemble for first hypercolumn
    n_timesteps = int(100 / SIM_DT)

    # Collect spike counts per ensemble
    n_ens = net.n_orientations
    neurons_per_ens = net.neurons_per_ensemble
    ensemble_spikes = np.zeros(n_ens)

    exc_spikes_total = 0
    inh_spikes_total = 0

    for t in range(n_timesteps):
        time_ms = t * SIM_DT

        # Generate stimulus
        on_spikes, off_spikes = grating.generate_spikes(time_ms, orientation)

        # Step network
        net.step(time_ms, on_spikes, off_spikes, learning=False)

        # Count spikes
        exc_spikes = np.array(net.v1_exc.spikes, dtype=bool)
        inh_spikes = np.array(net.v1_inh.spikes, dtype=bool)

        exc_spikes_total += np.sum(exc_spikes)
        inh_spikes_total += np.sum(inh_spikes)

        # Count per ensemble for hypercolumn 0
        for ens in range(n_ens):
            start_idx = ens * neurons_per_ens
            end_idx = start_idx + neurons_per_ens
            ensemble_spikes[ens] += np.sum(exc_spikes[start_idx:end_idx])

    # Calculate firing rates
    n_exc = net.v1_exc.n_neurons
    n_inh = net.v1_inh.n_neurons
    duration_s = (n_timesteps * SIM_DT) / 1000.0

    exc_rate = exc_spikes_total / (n_exc * duration_s)
    inh_rate = inh_spikes_total / (n_inh * duration_s)

    print(f"\nFiring Rates (100ms, {orientation} deg):")
    print(f"  Excitatory: {exc_rate:.1f} Hz")
    print(f"  Inhibitory: {inh_rate:.1f} Hz")

    # Ensemble diversity (hypercolumn 0)
    print(f"\nEnsemble responses (Hypercolumn 0):")
    for ens in range(n_ens):
        ori = GRATING_ORIENTATIONS[ens]
        print(f"  Ensemble {ens} (pref {ori:.0f}°): {ensemble_spikes[ens]:.0f} spikes")

    mean_spikes = np.mean(ensemble_spikes)
    std_spikes = np.std(ensemble_spikes)
    cv = std_spikes / mean_spikes if mean_spikes > 0 else 0

    print(f"\nDiversity metrics:")
    print(f"  Mean spikes: {mean_spikes:.1f}")
    print(f"  Std spikes: {std_spikes:.1f}")
    print(f"  CV (coefficient of variation): {cv:.3f}")

    # Check weight statistics
    w = np.array(net.get_lgn_v1_weights())
    nonzero = w[w > 0]
    print(f"\nLGN->V1 weight statistics:")
    print(f"  Non-zero weights: {len(nonzero)}")
    print(f"  Mean: {np.mean(nonzero):.4f}")
    print(f"  Std: {np.std(nonzero):.4f}")
    print(f"  Min: {np.min(nonzero):.4f}")
    print(f"  Max: {np.max(nonzero):.4f}")

    # Test response to different orientations
    print("\n" + "=" * 60)
    print("Testing response to all orientations (50ms each)...")

    responses = np.zeros((n_ens, len(GRATING_ORIENTATIONS)))

    for ori_idx, ori in enumerate(GRATING_ORIENTATIONS):
        # Reset network state
        net.v1_exc.v[:] = net.v1_exc.v_rest
        net.v1_exc.u[:] = net.v1_exc.b * net.v1_exc.v_rest
        net.v1_inh.v[:] = net.v1_inh.v_rest
        net.v1_inh.u[:] = net.v1_inh.b * net.v1_inh.v_rest

        n_ts = int(50 / SIM_DT)
        for t in range(n_ts):
            time_ms = t * SIM_DT
            on_spikes, off_spikes = grating.generate_spikes(time_ms, ori)
            net.step(time_ms, on_spikes, off_spikes, learning=False)

            exc_spikes = np.array(net.v1_exc.spikes, dtype=bool)
            for ens in range(n_ens):
                start_idx = ens * neurons_per_ens
                end_idx = start_idx + neurons_per_ens
                responses[ens, ori_idx] += np.sum(exc_spikes[start_idx:end_idx])

    print(f"\nTuning curves (spikes in 50ms) - Hypercolumn 0:")
    print("Ensemble | " + " | ".join([f"{o:5.1f}°" for o in GRATING_ORIENTATIONS]))
    print("-" * 80)
    for ens in range(n_ens):
        pref = GRATING_ORIENTATIONS[ens]
        vals = " | ".join([f"{responses[ens, i]:6.0f}" for i in range(len(GRATING_ORIENTATIONS))])
        print(f"   {ens} ({pref:5.1f}°) | {vals}")

    # Calculate OSI for each ensemble
    print(f"\nOrientation Selectivity Index (OSI):")
    for ens in range(n_ens):
        r = responses[ens, :]
        if np.sum(r) > 0:
            theta = np.deg2rad(2 * GRATING_ORIENTATIONS)  # Double for circular variance
            osi = np.abs(np.sum(r * np.exp(1j * theta))) / np.sum(r)
        else:
            osi = 0
        pref = GRATING_ORIENTATIONS[ens]
        print(f"  Ensemble {ens} (pref {pref:.0f}°): OSI = {osi:.3f}")

    print("\n" + "=" * 60)
    print("Diagnostic complete")

if __name__ == "__main__":
    main()
