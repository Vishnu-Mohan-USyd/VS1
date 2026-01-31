#!/usr/bin/env python3
"""Debug conductance values to understand inhibition problem."""

import numpy as np
import sys
sys.modules['cupy'] = None

from network import VisualCortexNetwork
from stimuli import DriftingGratingGenerator
from config import SIM_DT, E_AMPA, E_GABA

def main():
    print("Conductance Debug")
    print("=" * 60)

    net = VisualCortexNetwork()
    grating = DriftingGratingGenerator()

    # Check weight matrix shapes
    print(f"\nWeight matrix shapes:")
    print(f"  w_exc_inh: {net.w_exc_inh.shape}")
    print(f"  w_inh_exc: {net.w_inh_exc.shape}")
    print(f"  lgn_v1_weights: {net.lgn_v1_stdp.weights.shape}")

    print(f"\nWeight statistics:")
    print(f"  w_inh_exc - non-zero: {np.count_nonzero(net.w_inh_exc)}")
    print(f"  w_inh_exc - mean (non-zero): {np.mean(net.w_inh_exc[net.w_inh_exc > 0]):.4f}")
    print(f"  w_inh_exc - max: {np.max(net.w_inh_exc):.4f}")

    # Run a few timesteps and track conductances
    print(f"\n{'='*60}")
    print("Running simulation and tracking conductances...")

    orientation = 0.0

    for t_idx in range(20):  # Just 20 timesteps
        time_ms = t_idx * SIM_DT
        on_spikes, off_spikes = grating.generate_spikes(time_ms, orientation)

        # Store pre-step values
        g_exc_before = net.v1_exc.g_exc.copy()
        g_inh_before = net.v1_exc.g_inh.copy()
        v_before = net.v1_exc.v.copy()

        # Step
        spikes = net.step(time_ms, on_spikes, off_spikes, learning=False)

        # Get post-step values
        g_exc_after = net.v1_exc.g_exc.copy()
        g_inh_after = net.v1_exc.g_inh.copy()
        v_after = net.v1_exc.v.copy()

        # Calculate currents
        I_exc = g_exc_after * (E_AMPA - v_after)  # Should be positive (depolarizing)
        I_inh = g_inh_after * (E_GABA - v_after)  # Should be negative (hyperpolarizing)

        print(f"\nt={time_ms:.0f}ms:")
        print(f"  Spikes: LGN_ON={spikes['lgn_on']}, LGN_OFF={spikes['lgn_off']}, V1_Exc={spikes['v1_exc']}, V1_Inh={spikes['v1_inh']}")
        print(f"  g_exc: mean={np.mean(g_exc_after):.3f}, max={np.max(g_exc_after):.3f}")
        print(f"  g_inh: mean={np.mean(g_inh_after):.3f}, max={np.max(g_inh_after):.3f}")
        print(f"  I_exc: mean={np.mean(I_exc):.3f}, max={np.max(I_exc):.3f}")
        print(f"  I_inh: mean={np.mean(I_inh):.3f}, min={np.min(I_inh):.3f}")
        print(f"  V: mean={np.mean(v_after):.1f}, min={np.min(v_after):.1f}, max={np.max(v_after):.1f}")

if __name__ == "__main__":
    main()
