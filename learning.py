"""
Learning rules for the spiking neural network.
Implements STDP (Spike-Timing Dependent Plasticity) for LGN -> V1 connections.
"""

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

from config import (
    DT, USE_GPU,
    STDP_TAU_PLUS, STDP_TAU_MINUS,
    STDP_A_PLUS, STDP_A_MINUS,
    STDP_W_MAX, STDP_W_MIN,
    LGN_PATCH_SIZE, LGN_SIZE
)
from stimulus import get_array_module, to_numpy, to_device


class STDPLearner:
    """
    Spike-Timing Dependent Plasticity learning rule.

    Implements the classic STDP rule:
    - Pre before post (causal): weight increases (LTP)
    - Post before pre (anti-causal): weight decreases (LTD)

    Uses exponential eligibility traces for efficient computation.
    """

    def __init__(self, v1_layer, lgn_layer,
                 tau_plus=STDP_TAU_PLUS, tau_minus=STDP_TAU_MINUS,
                 a_plus=STDP_A_PLUS, a_minus=STDP_A_MINUS,
                 w_max=STDP_W_MAX, w_min=STDP_W_MIN,
                 dt=DT):
        """
        Initialize STDP learner.

        Args:
            v1_layer: V1 layer with weights to be trained
            lgn_layer: LGN layer (presynaptic)
            tau_plus: Time constant for potentiation trace (ms)
            tau_minus: Time constant for depression trace (ms)
            a_plus: Learning rate for potentiation
            a_minus: Learning rate for depression
            w_max: Maximum weight
            w_min: Minimum weight
            dt: Time step (ms)
        """
        self.xp = get_array_module()
        self.v1 = v1_layer
        self.lgn = lgn_layer

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        self.w_min = w_min
        self.dt = dt

        # Decay factors
        self.decay_plus = self.xp.exp(-dt / tau_plus)
        self.decay_minus = self.xp.exp(-dt / tau_minus)

        # Initialize eligibility traces
        self._init_traces()

    def _init_traces(self):
        """Initialize eligibility traces for STDP."""
        xp = self.xp

        # Pre-synaptic traces (LGN spikes)
        # Shape: (n_hypercolumns_y, n_hypercolumns_x, 2, patch_size, patch_size)
        pre_shape = (
            self.v1.n_hypercolumns_y, self.v1.n_hypercolumns_x,
            2, self.v1.patch_size, self.v1.patch_size
        )
        self.pre_trace = xp.zeros(pre_shape, dtype=xp.float32)

        # Post-synaptic traces (V1 spikes)
        # Shape: (n_hypercolumns_y, n_hypercolumns_x, n_ensembles, ensemble_size)
        post_shape = self.v1.shape
        self.post_trace = xp.zeros(post_shape, dtype=xp.float32)

    def reset(self):
        """Reset eligibility traces."""
        self.pre_trace.fill(0)
        self.post_trace.fill(0)

    def update(self, lgn_on_spikes, lgn_off_spikes, v1_spikes):
        """
        Update weights based on current spikes using STDP.

        Args:
            lgn_on_spikes: LGN ON channel spikes
            lgn_off_spikes: LGN OFF channel spikes
            v1_spikes: V1 layer spikes
        """
        xp = self.xp

        # Update pre-synaptic traces (decay + new spikes)
        self.pre_trace *= self.decay_plus

        # Add new LGN spikes to pre-traces (for each hypercolumn's patch)
        for hy in range(self.v1.n_hypercolumns_y):
            for hx in range(self.v1.n_hypercolumns_x):
                y_start = hy * self.v1.patch_size
                x_start = hx * self.v1.patch_size

                on_patch = lgn_on_spikes[y_start:y_start+self.v1.patch_size,
                                         x_start:x_start+self.v1.patch_size]
                off_patch = lgn_off_spikes[y_start:y_start+self.v1.patch_size,
                                           x_start:x_start+self.v1.patch_size]

                self.pre_trace[hy, hx, 0] += on_patch
                self.pre_trace[hy, hx, 1] += off_patch

        # Update post-synaptic traces (decay + new spikes)
        self.post_trace *= self.decay_minus
        self.post_trace += v1_spikes

        # Compute weight changes using STDP rule
        self._apply_stdp(v1_spikes)

    def _apply_stdp(self, v1_spikes):
        """
        Apply STDP weight updates (vectorized for speed).

        LTP: When post fires, strengthen connections from recently active pre
        LTD: When pre fires, weaken connections to recently active post
        """
        xp = self.xp

        # For each hypercolumn (still need loop but inner operations vectorized)
        for hy in range(self.v1.n_hypercolumns_y):
            for hx in range(self.v1.n_hypercolumns_x):
                # Get pre-synaptic trace for this hypercolumn's LGN patch
                pre_trace = self.pre_trace[hy, hx]  # (2, patch_y, patch_x)

                # LTP: For all neurons that spiked, strengthen based on pre_trace
                # v1_spikes[hy, hx] shape: (n_ensembles, ensemble_size)
                # We want to add pre_trace to weights where v1 spiked
                spiked = v1_spikes[hy, hx]  # (n_ensembles, ensemble_size)

                # Expand dimensions for broadcasting
                # spiked: (n_ensembles, ensemble_size) -> (n_ensembles, ensemble_size, 1, 1, 1)
                spiked_expanded = spiked[:, :, xp.newaxis, xp.newaxis, xp.newaxis]

                # pre_trace: (2, patch_y, patch_x) -> (1, 1, 2, patch_y, patch_x)
                pre_trace_expanded = pre_trace[xp.newaxis, xp.newaxis, :, :, :]

                # LTP update: add A+ * pre_trace where neuron spiked
                dw_ltp = self.a_plus * spiked_expanded * pre_trace_expanded
                self.v1.weights[hy, hx] += dw_ltp

                # LTD: For all pre neurons that spiked, depress based on post_trace
                y_start = hy * self.v1.patch_size
                x_start = hx * self.v1.patch_size

                # Get current LGN spikes for this patch
                lgn_on = self.lgn.on_neurons.spikes[
                    y_start:y_start+self.v1.patch_size,
                    x_start:x_start+self.v1.patch_size
                ]
                lgn_off = self.lgn.off_neurons.spikes[
                    y_start:y_start+self.v1.patch_size,
                    x_start:x_start+self.v1.patch_size
                ]

                # Stack LGN spikes: (2, patch_y, patch_x)
                lgn_spikes = xp.stack([lgn_on, lgn_off], axis=0)

                # Post trace: (n_ensembles, ensemble_size)
                post_trace = self.post_trace[hy, hx]

                # Expand dimensions for broadcasting
                # post_trace: (n_ensembles, ensemble_size) -> (n_ensembles, ensemble_size, 1, 1, 1)
                post_trace_expanded = post_trace[:, :, xp.newaxis, xp.newaxis, xp.newaxis]

                # lgn_spikes: (2, patch_y, patch_x) -> (1, 1, 2, patch_y, patch_x)
                lgn_spikes_expanded = lgn_spikes[xp.newaxis, xp.newaxis, :, :, :]

                # LTD update: subtract where pre spiked and post was recently active
                dw_ltd = -self.a_minus * post_trace_expanded * lgn_spikes_expanded
                self.v1.weights[hy, hx] += dw_ltd

        # Clip weights to bounds
        self.v1.weights = xp.clip(self.v1.weights, self.w_min, self.w_max)

    def get_weight_statistics(self):
        """Get statistics about current weights."""
        weights = to_numpy(self.v1.weights)

        return {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'min': np.min(weights),
            'max': np.max(weights),
            'sparsity': np.mean(weights < 0.1)
        }


class CompetitiveLearning:
    """
    Implements competitive learning through lateral inhibition and homeostasis.

    This helps ensure that different ensembles specialize for different orientations.
    """

    def __init__(self, v1_layer, target_rate=5.0, homeostasis_tau=10000.0, dt=DT):
        """
        Initialize competitive learning.

        Args:
            v1_layer: V1 layer
            target_rate: Target firing rate (Hz)
            homeostasis_tau: Time constant for homeostatic adaptation (ms)
            dt: Time step (ms)
        """
        self.xp = get_array_module()
        self.v1 = v1_layer
        self.target_rate = target_rate
        self.homeostasis_tau = homeostasis_tau
        self.dt = dt

        # Running estimate of firing rates for each ensemble
        self.rate_estimate = self.xp.ones(
            (v1_layer.n_hypercolumns_y, v1_layer.n_hypercolumns_x,
             v1_layer.n_ensembles),
            dtype=self.xp.float32
        ) * target_rate

        # Intrinsic excitability (homeostatic variable)
        self.excitability = self.xp.ones_like(self.rate_estimate)

        # Decay factor for rate estimation
        self.rate_decay = self.xp.exp(-dt / 1000.0)  # 1 second window

    def update(self, v1_spikes):
        """
        Update competitive learning based on current V1 activity.

        Args:
            v1_spikes: V1 spike array
        """
        xp = self.xp

        # Compute firing rate for each ensemble (spike count over neurons)
        ensemble_spikes = xp.sum(v1_spikes, axis=-1)  # (hy, hx, n_ensembles)

        # Update running rate estimate
        self.rate_estimate = (self.rate_estimate * self.rate_decay +
                             ensemble_spikes * (1 - self.rate_decay) * 1000 / self.dt)

        # Update excitability based on rate deviation from target
        rate_error = self.target_rate - self.rate_estimate
        excitability_update = rate_error * self.dt / self.homeostasis_tau
        self.excitability += excitability_update

        # Clip excitability to reasonable range
        self.excitability = xp.clip(self.excitability, 0.5, 2.0)

    def get_modulated_input(self, input_current):
        """
        Modulate input current based on homeostatic excitability.

        Args:
            input_current: Original input current to V1

        Returns:
            Modulated input current
        """
        xp = self.xp

        # Expand excitability to match input current shape
        excitability_expanded = self.excitability[:, :, :, xp.newaxis]

        return input_current * excitability_expanded

    def reset(self):
        """Reset competitive learning state."""
        self.rate_estimate.fill(self.target_rate)
        self.excitability.fill(1.0)


class OrientationSelectivityTracker:
    """
    Tracks the development of orientation selectivity during training.
    """

    def __init__(self, v1_layer, orientations):
        """
        Initialize tracker.

        Args:
            v1_layer: V1 layer to track
            orientations: Array of orientations being trained
        """
        self.v1 = v1_layer
        self.orientations = orientations
        self.n_orientations = len(orientations)

        # Response matrix: (n_hypercolumns_y, n_hypercolumns_x, n_ensembles, n_orientations)
        self.response_matrix = np.zeros((
            v1_layer.n_hypercolumns_y, v1_layer.n_hypercolumns_x,
            v1_layer.n_ensembles, self.n_orientations
        ))

        # History of selectivity over training
        self.selectivity_history = []

    def record_response(self, orientation_idx, spike_count):
        """
        Record V1 response to a specific orientation.

        Args:
            orientation_idx: Index of the orientation
            spike_count: Spike count for each ensemble (hy, hx, n_ensembles)
        """
        self.response_matrix[:, :, :, orientation_idx] = spike_count

    def compute_selectivity(self):
        """
        Compute orientation selectivity index (OSI) for each ensemble.

        OSI = (R_pref - R_orth) / (R_pref + R_orth)

        Returns:
            OSI array of shape (n_hypercolumns_y, n_hypercolumns_x, n_ensembles)
        """
        # Find preferred orientation (max response)
        pref_idx = np.argmax(self.response_matrix, axis=-1)

        # Get preferred response
        hy_idx, hx_idx, ens_idx = np.indices(pref_idx.shape)
        r_pref = self.response_matrix[hy_idx, hx_idx, ens_idx, pref_idx]

        # Get orthogonal orientation (90 degrees away)
        orth_offset = self.n_orientations // 2
        orth_idx = (pref_idx + orth_offset) % self.n_orientations
        r_orth = self.response_matrix[hy_idx, hx_idx, ens_idx, orth_idx]

        # Compute OSI
        osi = np.where(
            (r_pref + r_orth) > 0,
            (r_pref - r_orth) / (r_pref + r_orth + 1e-8),
            0
        )

        return osi

    def compute_preferred_orientations(self):
        """
        Compute preferred orientation for each ensemble.

        Returns:
            Array of preferred orientations in degrees
        """
        pref_idx = np.argmax(self.response_matrix, axis=-1)
        return self.orientations[pref_idx]

    def compute_tuning_width(self):
        """
        Compute tuning width (FWHM) for each ensemble.

        Returns:
            Array of tuning widths in degrees
        """
        # Simplified: use circular variance
        # Convert responses to complex numbers on unit circle
        theta = np.deg2rad(self.orientations * 2)  # Double angle for orientation

        # Normalize responses
        responses_norm = self.response_matrix / (np.sum(self.response_matrix, axis=-1, keepdims=True) + 1e-8)

        # Compute mean resultant vector
        real_part = np.sum(responses_norm * np.cos(theta), axis=-1)
        imag_part = np.sum(responses_norm * np.sin(theta), axis=-1)

        # Circular variance = 1 - |mean resultant|
        mean_resultant = np.sqrt(real_part**2 + imag_part**2)

        # Convert to approximate FWHM (narrower tuning = larger mean resultant)
        # This is a rough approximation
        tuning_width = 90 * (1 - mean_resultant)

        return tuning_width

    def save_snapshot(self):
        """Save current selectivity metrics to history."""
        self.selectivity_history.append({
            'osi': self.compute_selectivity().copy(),
            'preferred': self.compute_preferred_orientations().copy(),
            'tuning_width': self.compute_tuning_width().copy()
        })

    def get_summary_statistics(self):
        """Get summary statistics of orientation selectivity."""
        osi = self.compute_selectivity()
        pref = self.compute_preferred_orientations()
        width = self.compute_tuning_width()

        return {
            'mean_osi': np.mean(osi),
            'std_osi': np.std(osi),
            'mean_tuning_width': np.mean(width),
            'orientation_coverage': self._compute_coverage(pref)
        }

    def _compute_coverage(self, preferred_orientations):
        """Compute how well orientations are covered by ensembles."""
        # Flatten and compute histogram
        pref_flat = preferred_orientations.flatten()

        # Bin into orientation bins
        n_bins = len(self.orientations)
        bin_edges = np.concatenate([self.orientations, [180]])

        hist, _ = np.histogram(pref_flat, bins=bin_edges)

        # Coverage = fraction of bins with at least one ensemble
        coverage = np.sum(hist > 0) / n_bins

        return coverage


if __name__ == "__main__":
    # Test learning rules
    import matplotlib.pyplot as plt
    from layers import LGNLayer, V1Layer

    print("Testing learning rules...")

    # Create layers
    lgn = LGNLayer()
    v1 = V1Layer()

    # Create learner
    stdp = STDPLearner(v1, lgn)

    # Simulate some activity
    n_steps = 100

    for t in range(n_steps):
        # Random LGN spikes
        lgn_on = (np.random.random((LGN_SIZE, LGN_SIZE)) < 0.05).astype(np.float32)
        lgn_off = (np.random.random((LGN_SIZE, LGN_SIZE)) < 0.05).astype(np.float32)

        lgn_on = to_device(lgn_on)
        lgn_off = to_device(lgn_off)

        # Dummy V1 spikes
        v1_spikes = (np.random.random(v1.shape) < 0.01).astype(np.float32)
        v1_spikes = to_device(v1_spikes)

        # Update STDP
        stdp.update(lgn_on, lgn_off, v1_spikes)

    # Get weight statistics
    stats = stdp.get_weight_statistics()
    print(f"\nWeight statistics after {n_steps} steps:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Plot weight changes
    weights_after = v1.get_weights_for_hypercolumn(0, 0)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for ens in range(8):
        w_on = np.mean(weights_after[ens, :, 0], axis=0)
        w_off = np.mean(weights_after[ens, :, 1], axis=0)

        axes[0, ens].imshow(w_on, cmap='Reds', vmin=0, vmax=1)
        axes[0, ens].set_title(f'Ens {ens} ON')
        axes[0, ens].axis('off')

        axes[1, ens].imshow(w_off, cmap='Blues', vmin=0, vmax=1)
        axes[1, ens].set_title(f'Ens {ens} OFF')
        axes[1, ens].axis('off')

    plt.suptitle('Weights after random STDP updates')
    plt.tight_layout()
    plt.savefig('learning_test.png', dpi=100)
    plt.close()

    print("\nSaved weight visualization to learning_test.png")
