"""
Main network architecture: RGC -> LGN -> V1 Layer 4

Implements the full visual pathway with:
- RGC: ON and OFF channels (input layer, driven by stimuli)
- LGN: Relay neurons with ON/OFF channels
- V1 L4: Hypercolumns with orientation-selective ensembles
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
except ImportError:
    import numpy as np
    GPU_AVAILABLE = False
    xp = np

import numpy as np
from neurons import IzhikevichPopulation, SpikeDelayBuffer, to_numpy, to_gpu
from plasticity import STDP, InhibitorySTDP
from config import (
    RETINA_SIZE, LGN_PATCH_SIZE, HYPERCOLUMN_STRIDE,
    N_HYPERCOLUMNS_X, N_HYPERCOLUMNS_Y, N_HYPERCOLUMNS,
    N_ORIENTATIONS, NEURONS_PER_ENSEMBLE, INHIBITORY_PER_HYPERCOLUMN,
    W_RGC_LGN, W_LGN_V1_INIT_MEAN, W_LGN_V1_INIT_STD,
    W_LGN_V1_MIN, W_LGN_V1_MAX,
    DELAY_RGC_LGN, DELAY_LGN_V1_MIN, DELAY_LGN_V1_MAX,
    W_V1_EXC_LOCAL, W_V1_INH,
    P_LGN_V1, P_V1_LATERAL_EXC, P_V1_LATERAL_INH,
    SIM_DT
)


class VisualCortexNetwork:
    """
    Complete RGC -> LGN -> V1 L4 network.

    Architecture:
    - RGC: Direct spike input (not simulated neurons, just spikes from stimuli)
    - LGN: Relay neurons (ON and OFF populations)
    - V1: Hypercolumns containing orientation-selective ensembles

    Each hypercolumn receives input from a 10x10 LGN patch and contains
    8 orientation ensembles that compete via lateral inhibition.
    """

    def __init__(self):
        """Initialize the complete network."""
        print("=" * 60)
        print("Initializing Visual Cortex Network")
        print("=" * 60)

        # Network dimensions
        self.retina_size = RETINA_SIZE
        self.lgn_patch_size = LGN_PATCH_SIZE
        self.n_hypercolumns_x = N_HYPERCOLUMNS_X
        self.n_hypercolumns_y = N_HYPERCOLUMNS_Y
        self.n_hypercolumns = N_HYPERCOLUMNS
        self.n_orientations = N_ORIENTATIONS
        self.neurons_per_ensemble = NEURONS_PER_ENSEMBLE

        print(f"Retina/LGN size: {self.retina_size}x{self.retina_size}")
        print(f"LGN patch size: {self.lgn_patch_size}x{self.lgn_patch_size}")
        print(f"Hypercolumns: {self.n_hypercolumns_x}x{self.n_hypercolumns_y} = {self.n_hypercolumns}")
        print(f"Orientations per hypercolumn: {self.n_orientations}")
        print(f"Neurons per ensemble: {self.neurons_per_ensemble}")

        # Initialize populations
        self._init_lgn()
        self._init_v1()

        # Initialize connectivity
        self._init_lgn_v1_connections()
        self._init_v1_lateral_connections()

        # Delay buffers
        n_lgn = self.retina_size * self.retina_size * 2  # ON + OFF
        self.lgn_delay_buffer = SpikeDelayBuffer(n_lgn, DELAY_LGN_V1_MAX)

        print("=" * 60)
        print("Network initialization complete")
        print("=" * 60)

    def _init_lgn(self):
        """Initialize LGN populations (ON and OFF channels)."""
        n_lgn_per_channel = self.retina_size * self.retina_size

        # LGN ON cells - thalamocortical relay neurons
        self.lgn_on = IzhikevichPopulation(
            n_lgn_per_channel, neuron_type='TC', name='LGN_ON'
        )

        # LGN OFF cells
        self.lgn_off = IzhikevichPopulation(
            n_lgn_per_channel, neuron_type='TC', name='LGN_OFF'
        )

        # Strong feedforward weights from RGC to LGN (one-to-one)
        self.w_rgc_lgn = W_RGC_LGN

        print(f"LGN: {n_lgn_per_channel} ON + {n_lgn_per_channel} OFF neurons")

    def _init_v1(self):
        """Initialize V1 Layer 4 populations."""
        # Total excitatory neurons in V1
        self.n_v1_exc = self.n_hypercolumns * self.n_orientations * self.neurons_per_ensemble

        # Total inhibitory neurons in V1
        self.n_v1_inh = self.n_hypercolumns * INHIBITORY_PER_HYPERCOLUMN

        # Excitatory population (Regular Spiking)
        self.v1_exc = IzhikevichPopulation(
            self.n_v1_exc, neuron_type='RS', name='V1_Exc'
        )

        # Inhibitory population (Fast Spiking)
        self.v1_inh = IzhikevichPopulation(
            self.n_v1_inh, neuron_type='FS', name='V1_Inh'
        )

        print(f"V1 Excitatory: {self.n_v1_exc} neurons")
        print(f"V1 Inhibitory: {self.n_v1_inh} neurons")

        # Create mapping from (hypercolumn, orientation, neuron) to flat index
        self._create_v1_indexing()

    def _create_v1_indexing(self):
        """Create index mappings for V1 neurons."""
        # Excitatory indexing: (hypercolumn_idx, orientation_idx, neuron_idx) -> flat_idx
        self.exc_idx = np.zeros(
            (self.n_hypercolumns, self.n_orientations, self.neurons_per_ensemble),
            dtype=np.int32
        )

        idx = 0
        for hc in range(self.n_hypercolumns):
            for ori in range(self.n_orientations):
                for n in range(self.neurons_per_ensemble):
                    self.exc_idx[hc, ori, n] = idx
                    idx += 1

        # Inhibitory indexing: (hypercolumn_idx, neuron_idx) -> flat_idx
        self.inh_idx = np.zeros(
            (self.n_hypercolumns, INHIBITORY_PER_HYPERCOLUMN),
            dtype=np.int32
        )

        idx = 0
        for hc in range(self.n_hypercolumns):
            for n in range(INHIBITORY_PER_HYPERCOLUMN):
                self.inh_idx[hc, n] = idx
                idx += 1

        # Hypercolumn center positions (in retina coordinates)
        self.hc_centers = np.zeros((self.n_hypercolumns, 2), dtype=np.int32)
        hc_idx = 0
        for hx in range(self.n_hypercolumns_x):
            for hy in range(self.n_hypercolumns_y):
                cx = hx * HYPERCOLUMN_STRIDE + self.lgn_patch_size // 2
                cy = hy * HYPERCOLUMN_STRIDE + self.lgn_patch_size // 2
                self.hc_centers[hc_idx] = [cx, cy]
                hc_idx += 1

    def _init_lgn_v1_connections(self):
        """
        Initialize LGN -> V1 connections.

        Each hypercolumn receives input from a 10x10 LGN patch.
        Each orientation ensemble within the hypercolumn receives
        the same patch but with different initial weights and delays.
        """
        print("Initializing LGN -> V1 connections...")

        n_lgn = self.retina_size * self.retina_size * 2  # ON + OFF combined
        n_v1_exc = self.n_v1_exc

        # Initialize weight matrix (sparse, but we use dense for GPU efficiency)
        # Rows: LGN neurons (ON then OFF)
        # Cols: V1 excitatory neurons
        weights = np.zeros((n_lgn, n_v1_exc), dtype=np.float32)

        # Initialize delay matrix (same shape)
        self.lgn_v1_delays = np.zeros((n_lgn, n_v1_exc), dtype=np.float32)

        # For each hypercolumn
        for hc_idx in range(self.n_hypercolumns):
            cx, cy = self.hc_centers[hc_idx]

            # Get LGN patch indices
            x_start = cx - self.lgn_patch_size // 2
            y_start = cy - self.lgn_patch_size // 2

            # LGN indices for this patch (both ON and OFF)
            lgn_on_indices = []
            lgn_off_indices = []

            for dx in range(self.lgn_patch_size):
                for dy in range(self.lgn_patch_size):
                    x = x_start + dx
                    y = y_start + dy

                    if 0 <= x < self.retina_size and 0 <= y < self.retina_size:
                        lgn_idx = y * self.retina_size + x
                        lgn_on_indices.append(lgn_idx)
                        lgn_off_indices.append(lgn_idx + self.retina_size * self.retina_size)

            lgn_patch_indices = lgn_on_indices + lgn_off_indices

            # For each orientation ensemble in this hypercolumn
            for ori_idx in range(self.n_orientations):
                # Get V1 neuron indices for this ensemble
                v1_indices = self.exc_idx[hc_idx, ori_idx, :].flatten()

                # Create connections from LGN patch to this ensemble
                # NO artificial bias - purely random initial weights
                # Orientation selectivity must emerge from STDP + competition
                for lgn_idx in lgn_patch_indices:
                    for v1_idx in v1_indices:
                        # Random connection probability
                        if np.random.random() < P_LGN_V1:
                            # Random initial weight
                            w = W_LGN_V1_INIT_MEAN + W_LGN_V1_INIT_STD * np.random.randn()
                            w = np.clip(w, W_LGN_V1_MIN + 0.001, W_LGN_V1_MAX)
                            weights[lgn_idx, v1_idx] = w

                            # Random delay (some variation to break symmetry)
                            delay = DELAY_LGN_V1_MIN + np.random.random() * (DELAY_LGN_V1_MAX - DELAY_LGN_V1_MIN)
                            self.lgn_v1_delays[lgn_idx, v1_idx] = delay

        # Create STDP object for these connections
        self.lgn_v1_stdp = STDP(n_lgn, n_v1_exc, weights,
                                w_min=W_LGN_V1_MIN, w_max=W_LGN_V1_MAX)

        n_connections = np.sum(weights > 0)
        print(f"LGN -> V1: {n_connections} connections")

        # Convert delays to GPU
        self.lgn_v1_delays = to_gpu(self.lgn_v1_delays)

        # Pre-compute average delays per LGN neuron for efficiency
        # Use a single average delay for all LGN neurons
        delay_mask = weights > 0
        if np.sum(delay_mask) > 0:
            avg_delay = np.mean(self.lgn_v1_delays[delay_mask].get() if GPU_AVAILABLE else self.lgn_v1_delays[delay_mask])
        else:
            avg_delay = DELAY_LGN_V1_MIN
        self.avg_lgn_v1_delays = to_gpu(np.full(n_lgn, avg_delay, dtype=np.float32))
        print(f"Average LGN-V1 delay: {avg_delay:.2f} ms")

    def _init_v1_lateral_connections(self):
        """
        Initialize lateral connections within V1.

        - Excitatory connections: Between neurons with similar orientation preference
        - Inhibitory connections: Between neurons with different orientation preferences
          (especially orthogonal) to create competition
        """
        print("Initializing V1 lateral connections...")

        # Excitatory lateral connections (E -> E, E -> I)
        # Within hypercolumn, similar orientations excite each other

        # E -> E weights
        self.w_exc_exc = np.zeros((self.n_v1_exc, self.n_v1_exc), dtype=np.float32)

        # E -> I weights (excitatory neurons drive inhibitory)
        self.w_exc_inh = np.zeros((self.n_v1_exc, self.n_v1_inh), dtype=np.float32)

        # I -> E weights (inhibitory neurons suppress excitatory)
        self.w_inh_exc = np.zeros((self.n_v1_inh, self.n_v1_exc), dtype=np.float32)

        # For each hypercolumn - implement winner-take-all through uniform lateral inhibition
        # NO HARDCODING of orientation preferences!
        for hc_idx in range(self.n_hypercolumns):
            # Get all neurons in this hypercolumn
            exc_in_hc = self.exc_idx[hc_idx, :, :].flatten()
            inh_in_hc = self.inh_idx[hc_idx, :].flatten()

            # E -> E: NO direct excitatory connections between ensembles
            # This forces competition - ensembles must differentiate through STDP alone
            # (Remove lateral excitation to make competition cleaner)

            # E -> I: ALL excitatory neurons drive inhibitory neurons (uniform)
            for e_idx in exc_in_hc:
                for i_idx in inh_in_hc:
                    if np.random.random() < P_V1_LATERAL_INH:
                        self.w_exc_inh[e_idx, i_idx] = W_V1_EXC_LOCAL * (0.8 + 0.4 * np.random.random())

            # I -> E: ALL inhibitory neurons suppress ALL excitatory neurons (uniform)
            # This creates winner-take-all: most active ensemble suppresses others
            for i_idx in inh_in_hc:
                for e_idx in exc_in_hc:
                    if np.random.random() < P_V1_LATERAL_INH:
                        self.w_inh_exc[i_idx, e_idx] = W_V1_INH * (0.8 + 0.4 * np.random.random())

        # Convert to GPU
        self.w_exc_exc = to_gpu(self.w_exc_exc)
        self.w_exc_inh = to_gpu(self.w_exc_inh)
        self.w_inh_exc = to_gpu(self.w_inh_exc)

        n_ee = np.sum(to_numpy(self.w_exc_exc) > 0)
        n_ei = np.sum(to_numpy(self.w_exc_inh) > 0)
        n_ie = np.sum(to_numpy(self.w_inh_exc) > 0)
        print(f"V1 E->E: {n_ee}, E->I: {n_ei}, I->E: {n_ie} connections")

        # Initialize inhibitory STDP for I->E connections
        self.inh_stdp = InhibitorySTDP(
            self.n_v1_inh, self.n_v1_exc,
            to_numpy(self.w_inh_exc),
            target_rate=15.0
        )

    def step(self, t, on_spikes, off_spikes, learning=True, dt=SIM_DT):
        """
        Advance simulation by one time step.

        Args:
            t: Current time (ms)
            on_spikes: ON channel spikes from stimulus (retina_size x retina_size)
            off_spikes: OFF channel spikes from stimulus
            learning: Whether to update synaptic weights
            dt: Time step (ms)

        Returns:
            Dictionary of spike counts for each population
        """
        # Flatten RGC spikes
        on_flat = on_spikes.flatten() if len(on_spikes.shape) > 1 else on_spikes
        off_flat = off_spikes.flatten() if len(off_spikes.shape) > 1 else off_spikes

        # === LGN Processing ===
        # Drive LGN with RGC spikes (strong feedforward)
        self.lgn_on.I_ext = self.w_rgc_lgn * on_flat.astype(xp.float32)
        self.lgn_off.I_ext = self.w_rgc_lgn * off_flat.astype(xp.float32)

        # Step LGN
        lgn_on_spikes = self.lgn_on.step(t, dt)
        lgn_off_spikes = self.lgn_off.step(t, dt)

        # Combine LGN spikes
        lgn_spikes = xp.concatenate([lgn_on_spikes, lgn_off_spikes])

        # === Delay Buffer ===
        self.lgn_delay_buffer.add_spikes(lgn_spikes)

        # === V1 Processing ===
        # Get delayed LGN spikes (using pre-computed average delay)
        delayed_lgn_spikes = self.lgn_delay_buffer.get_delayed_spikes(
            self.avg_lgn_v1_delays
        )

        # Feedforward input to V1 excitatory
        weights = self.lgn_v1_stdp.weights
        self.v1_exc.receive_spikes(weights, delayed_lgn_spikes, excitatory=True)

        # V1 lateral connections
        v1_exc_spikes = self.v1_exc.spikes

        # E -> E and E -> I
        if xp.any(v1_exc_spikes):
            self.v1_exc.receive_spikes(self.w_exc_exc, v1_exc_spikes, excitatory=True)
            self.v1_inh.receive_spikes(self.w_exc_inh, v1_exc_spikes, excitatory=True)

        # Step V1 populations
        v1_exc_spikes = self.v1_exc.step(t, dt)
        v1_inh_spikes = self.v1_inh.step(t, dt)

        # I -> E
        if xp.any(v1_inh_spikes):
            self.v1_exc.receive_spikes(self.w_inh_exc, v1_inh_spikes, excitatory=False)

        # === Learning (STDP) ===
        if learning:
            # LGN -> V1 excitatory STDP
            self.lgn_v1_stdp.update(delayed_lgn_spikes, v1_exc_spikes, dt)

            # Inhibitory STDP for homeostasis
            self.inh_stdp.update(v1_inh_spikes, v1_exc_spikes, dt, learning_rate=0.5)
            self.w_inh_exc = self.inh_stdp.weights

        # Advance delay buffer
        self.lgn_delay_buffer.step()

        # Return spike counts
        return {
            'lgn_on': int(to_numpy(xp.sum(lgn_on_spikes))),
            'lgn_off': int(to_numpy(xp.sum(lgn_off_spikes))),
            'v1_exc': int(to_numpy(xp.sum(v1_exc_spikes))),
            'v1_inh': int(to_numpy(xp.sum(v1_inh_spikes))),
        }

    def reset(self):
        """Reset all neuron states (but keep learned weights)."""
        self.lgn_on.reset()
        self.lgn_off.reset()
        self.v1_exc.reset()
        self.v1_inh.reset()
        self.lgn_delay_buffer.reset()
        self.lgn_v1_stdp.reset_traces()
        self.inh_stdp.reset_traces()

    def get_lgn_v1_weights(self):
        """Get LGN -> V1 weight matrix as numpy array."""
        return self.lgn_v1_stdp.get_weights()

    def get_ensemble_weights(self, hypercolumn_idx, orientation_idx):
        """
        Get total input weights for a specific ensemble.

        Args:
            hypercolumn_idx: Index of hypercolumn
            orientation_idx: Index of orientation within hypercolumn

        Returns:
            Total weight from LGN patch to this ensemble
        """
        v1_indices = self.exc_idx[hypercolumn_idx, orientation_idx, :].flatten()
        weights = self.lgn_v1_stdp.get_weights()

        # Sum weights for all neurons in ensemble
        total_weights = np.sum(weights[:, v1_indices], axis=1)

        return total_weights

    def get_all_ensemble_weights(self):
        """
        Get weight sums for all ensembles.

        Returns:
            Array of shape (n_hypercolumns, n_orientations) with total input weights
        """
        weights = self.lgn_v1_stdp.get_weights()
        ensemble_weights = np.zeros((self.n_hypercolumns, self.n_orientations))

        for hc in range(self.n_hypercolumns):
            for ori in range(self.n_orientations):
                v1_indices = self.exc_idx[hc, ori, :].flatten()
                ensemble_weights[hc, ori] = np.sum(weights[:, v1_indices])

        return ensemble_weights
