"""
Spiking Neural Network for Orientation Selectivity Emergence
RGC (ON/OFF) -> LGN -> V1 Layer 4

This implements the hypothesis that orientation selectivity emerges from:
1. Retinotopic LGN patches projecting to multiple V1 ensembles
2. STDP-based learning selecting appropriate connections
3. Lateral competition ensuring diverse orientation preferences

Uses Izhikevich neurons with biologically realistic parameters.
GPU-accelerated with CuPy (falls back to NumPy if unavailable).

Author: Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from typing import Tuple, Dict, List, Optional
import time
import os

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy available - using GPU acceleration")
except ImportError:
    cp = np  # Fallback to numpy
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU (NumPy)")


def to_numpy(arr):
    """Convert array to numpy (handles both cupy and numpy arrays)."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.array(arr)


# ==============================================================================
# IZHIKEVICH NEURON PARAMETERS
# Based on Izhikevich (2003) "Simple Model of Spiking Neurons" and
# Izhikevich (2004) "Which Model to Use for Cortical Spiking Neurons?"
# ==============================================================================

# Regular Spiking (RS) - Typical excitatory cortical neurons
IZHIKEVICH_RS = {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}

# Fast Spiking (FS) - Inhibitory interneurons (basket cells)
IZHIKEVICH_FS = {'a': 0.1, 'b': 0.2, 'c': -65.0, 'd': 2.0}

# Thalamic relay neurons (TC) - For LGN
# Parameters from Izhikevich (2003) for thalamic cells
IZHIKEVICH_TC = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05}

# Retinal Ganglion Cells - Similar to RS but faster
IZHIKEVICH_RGC = {'a': 0.03, 'b': 0.25, 'c': -65.0, 'd': 4.0}


class IzhikevichPopulation:
    """
    Population of Izhikevich neurons with GPU acceleration.

    The Izhikevich model:
    dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    du/dt = a*(b*v - u)
    if v >= 30 mV: v <- c, u <- u + d
    """

    def __init__(self, n_neurons: int, params: dict, name: str = ""):
        self.n = n_neurons
        self.name = name
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']

        # State variables
        self.v = cp.ones(n_neurons) * -65.0  # Membrane potential
        self.u = cp.ones(n_neurons) * self.b * (-65.0)  # Recovery variable
        self.fired = cp.zeros(n_neurons, dtype=bool)  # Spike indicator

    def step(self, I: cp.ndarray, dt: float = 0.5) -> cp.ndarray:
        """
        Advance the neuron population by one timestep.

        Args:
            I: Input current to each neuron
            dt: Timestep in ms (use 0.5ms for stability with Euler)

        Returns:
            Boolean array indicating which neurons fired
        """
        # Two 0.5ms sub-steps for stability (Izhikevich's recommendation)
        for _ in range(2):
            # Voltage dynamics
            dv = (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + I) * (dt/2)
            self.v = self.v + dv

            # Recovery dynamics
            du = self.a * (self.b * self.v - self.u) * (dt/2)
            self.u = self.u + du

        # Check for spikes
        self.fired = self.v >= 30.0

        # Reset neurons that fired
        self.v = cp.where(self.fired, self.c, self.v)
        self.u = cp.where(self.fired, self.u + self.d, self.u)

        return self.fired

    def reset(self):
        """Reset all neurons to resting state."""
        self.v = cp.ones(self.n) * -65.0
        self.u = cp.ones(self.n) * self.b * (-65.0)
        self.fired = cp.zeros(self.n, dtype=bool)


class STDPSynapse:
    """
    Spike-Timing Dependent Plasticity synapse with eligibility traces.

    Uses the classic STDP rule:
    - Pre before post (causal): potentiation
    - Post before pre (acausal): depression
    """

    def __init__(self, n_pre: int, n_post: int,
                 connectivity: cp.ndarray,
                 w_init: cp.ndarray,
                 delays: cp.ndarray,
                 w_max: float = 1.0,
                 w_min: float = 0.0,
                 tau_pre: float = 20.0,
                 tau_post: float = 20.0,
                 A_plus: float = 0.005,
                 A_minus: float = 0.005,
                 max_delay: int = 20):
        """
        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            connectivity: Boolean matrix (n_pre x n_post) indicating connections
            w_init: Initial weights (n_pre x n_post)
            delays: Conduction delays in timesteps (n_pre x n_post)
            w_max: Maximum weight
            w_min: Minimum weight
            tau_pre/tau_post: STDP time constants in ms
            A_plus/A_minus: STDP learning rates
            max_delay: Maximum delay in timesteps for delay buffer
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.connectivity = connectivity
        self.weights = w_init * connectivity
        self.delays = delays.astype(cp.int32)
        self.max_delay = max_delay

        self.w_max = w_max
        self.w_min = w_min
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_plus = A_plus
        self.A_minus = A_minus

        # Eligibility traces
        self.trace_pre = cp.zeros(n_pre)
        self.trace_post = cp.zeros(n_post)

        # Spike delay buffer (circular buffer)
        self.spike_buffer = cp.zeros((max_delay + 1, n_pre), dtype=bool)
        self.buffer_idx = 0

    def propagate(self, pre_fired: cp.ndarray) -> cp.ndarray:
        """
        Propagate spikes through synapses with delays.

        Returns:
            Current delivered to each postsynaptic neuron
        """
        # Store current spikes in buffer
        self.spike_buffer[self.buffer_idx] = pre_fired

        # Calculate which presynaptic spikes arrive now (based on delays)
        post_current = cp.zeros(self.n_post)

        # For each unique delay value, gather spikes and compute contribution
        unique_delays = cp.unique(self.delays[self.connectivity])

        for delay in unique_delays:
            delay_int = int(delay)
            # Get spikes from 'delay' timesteps ago
            buffer_read_idx = (self.buffer_idx - delay_int) % (self.max_delay + 1)
            delayed_spikes = self.spike_buffer[buffer_read_idx]

            # Find connections with this delay
            delay_mask = (self.delays == delay_int) & self.connectivity

            # Add current contribution
            # Weight * spike for all matching connections
            spike_matrix = cp.outer(delayed_spikes, cp.ones(self.n_post))
            contribution = cp.sum(self.weights * delay_mask * spike_matrix, axis=0)
            post_current += contribution

        # Advance buffer index
        self.buffer_idx = (self.buffer_idx + 1) % (self.max_delay + 1)

        return post_current

    def update_stdp(self, pre_fired: cp.ndarray, post_fired: cp.ndarray, dt: float = 1.0):
        """
        Update weights based on STDP rule.
        """
        # Update traces (exponential decay)
        decay_pre = cp.exp(-dt / self.tau_pre)
        decay_post = cp.exp(-dt / self.tau_post)

        self.trace_pre = self.trace_pre * decay_pre
        self.trace_post = self.trace_post * decay_post

        # Increment traces for neurons that fired
        self.trace_pre = cp.where(pre_fired, self.trace_pre + 1.0, self.trace_pre)
        self.trace_post = cp.where(post_fired, self.trace_post + 1.0, self.trace_post)

        # STDP weight updates
        # When post fires: potentiate synapses from recently active pre neurons
        if cp.any(post_fired):
            # For each postsynaptic neuron that fired, increase weights from pre neurons with high trace
            dw_plus = self.A_plus * cp.outer(self.trace_pre, post_fired.astype(cp.float64))
            self.weights = self.weights + dw_plus * self.connectivity

        # When pre fires: depress synapses to recently active post neurons
        if cp.any(pre_fired):
            # For each presynaptic neuron that fired, decrease weights to post neurons with high trace
            dw_minus = self.A_minus * cp.outer(pre_fired.astype(cp.float64), self.trace_post)
            self.weights = self.weights - dw_minus * self.connectivity

        # Clamp weights
        self.weights = cp.clip(self.weights, self.w_min, self.w_max)

    def reset(self):
        """Reset traces and spike buffer."""
        self.trace_pre = cp.zeros(self.n_pre)
        self.trace_post = cp.zeros(self.n_post)
        self.spike_buffer = cp.zeros((self.max_delay + 1, self.n_pre), dtype=bool)
        self.buffer_idx = 0


class DriftingGratingGenerator:
    """
    Generates drifting sinusoidal grating stimuli.

    Creates ON and OFF channel responses for a given orientation and
    spatial/temporal frequency.
    """

    def __init__(self, width: int, height: int,
                 spatial_freq: float = 0.1,  # cycles per pixel
                 temporal_freq: float = 2.0,  # Hz
                 contrast: float = 1.0):
        self.width = width
        self.height = height
        self.spatial_freq = spatial_freq
        self.temporal_freq = temporal_freq
        self.contrast = contrast

        # Create coordinate grids
        x = np.arange(width)
        y = np.arange(height)
        self.X, self.Y = np.meshgrid(x, y)

    def generate(self, orientation: float, time_ms: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ON and OFF channel responses for a drifting grating.

        Args:
            orientation: Grating orientation in degrees (0 = vertical, 90 = horizontal)
            time_ms: Current time in milliseconds

        Returns:
            (on_response, off_response): Tuple of 2D arrays with firing rates
        """
        # Convert orientation to radians
        theta = np.radians(orientation)

        # Compute the grating
        # Direction perpendicular to orientation for drift
        phase = 2 * np.pi * self.temporal_freq * (time_ms / 1000.0)

        # Spatial frequency component along grating direction
        spatial_phase = 2 * np.pi * self.spatial_freq * (
            self.X * np.cos(theta) + self.Y * np.sin(theta)
        )

        # Sinusoidal grating (ranges from -1 to 1)
        grating = self.contrast * np.sin(spatial_phase + phase)

        # Split into ON and OFF channels
        # ON cells respond to positive values (light increments)
        # OFF cells respond to negative values (light decrements)
        on_response = np.maximum(0, grating)
        off_response = np.maximum(0, -grating)

        return on_response, off_response

    def generate_spike_rates(self, orientation: float, time_ms: float,
                            base_rate: float = 5.0,
                            max_rate: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert grating to spike rates for Poisson spike generation.

        Args:
            base_rate: Baseline firing rate in Hz
            max_rate: Maximum firing rate in Hz

        Returns:
            (on_rates, off_rates): Firing rates in Hz for each cell
        """
        on_resp, off_resp = self.generate(orientation, time_ms)

        # Convert to firing rates
        on_rates = base_rate + (max_rate - base_rate) * on_resp
        off_rates = base_rate + (max_rate - base_rate) * off_resp

        return on_rates, off_rates


class OrientationSelectivityNetwork:
    """
    Complete network implementing the RGC -> LGN -> V1 pathway
    with orientation selectivity emergence through STDP.
    """

    def __init__(self,
                 retina_size: Tuple[int, int] = (16, 16),
                 lgn_patch_size: int = 4,
                 n_orientations: int = 8,
                 max_delay: int = 15,
                 stdp_params: Optional[dict] = None):
        """
        Args:
            retina_size: (width, height) of retinal/LGN arrays
            lgn_patch_size: Size of each retinotopic patch (e.g., 4 for 4x4)
            n_orientations: Number of V1 orientation ensembles per patch
            max_delay: Maximum conduction delay in ms
            stdp_params: Dictionary of STDP parameters
        """
        self.retina_width, self.retina_height = retina_size
        self.patch_size = lgn_patch_size
        self.n_orientations = n_orientations
        self.max_delay = max_delay

        # Calculate grid of patches
        self.n_patches_x = self.retina_width // lgn_patch_size
        self.n_patches_y = self.retina_height // lgn_patch_size
        self.n_patches = self.n_patches_x * self.n_patches_y

        # Total neurons in each layer
        self.n_rgc = self.retina_width * self.retina_height  # Per channel (ON/OFF)
        self.n_lgn = self.n_rgc  # Same size, per channel
        self.n_v1_exc = self.n_patches * n_orientations  # Excitatory V1 neurons
        self.n_v1_inh = self.n_patches * (n_orientations // 2)  # Inhibitory (fewer)

        print(f"Network Architecture:")
        print(f"  Retina: {retina_size[0]}x{retina_size[1]} (ON + OFF = {2*self.n_rgc} cells)")
        print(f"  LGN: {retina_size[0]}x{retina_size[1]} (ON + OFF = {2*self.n_lgn} cells)")
        print(f"  V1 Patches: {self.n_patches_x}x{self.n_patches_y} = {self.n_patches}")
        print(f"  V1 Excitatory: {self.n_v1_exc} ({n_orientations} per patch)")
        print(f"  V1 Inhibitory: {self.n_v1_inh}")

        # Default STDP parameters
        self.stdp_params = stdp_params or {
            'tau_pre': 20.0,
            'tau_post': 20.0,
            'A_plus': 0.01,
            'A_minus': 0.012,  # Slightly stronger depression for competition
            'w_max': 1.0,
            'w_min': 0.0
        }

        # Initialize neuron populations
        self._init_neurons()

        # Initialize synapses
        self._init_synapses()

        # Stimulus generator
        self.stim_gen = DriftingGratingGenerator(
            self.retina_width, self.retina_height,
            spatial_freq=0.15,
            temporal_freq=2.0
        )

        # Recording variables
        self.spike_history = {
            'rgc_on': [], 'rgc_off': [],
            'lgn_on': [], 'lgn_off': [],
            'v1_exc': [], 'v1_inh': []
        }
        self.weight_history = []

    def _init_neurons(self):
        """Initialize all neuron populations."""
        # RGC populations (ON and OFF)
        self.rgc_on = IzhikevichPopulation(self.n_rgc, IZHIKEVICH_RGC, "RGC_ON")
        self.rgc_off = IzhikevichPopulation(self.n_rgc, IZHIKEVICH_RGC, "RGC_OFF")

        # LGN populations (ON and OFF)
        self.lgn_on = IzhikevichPopulation(self.n_lgn, IZHIKEVICH_TC, "LGN_ON")
        self.lgn_off = IzhikevichPopulation(self.n_lgn, IZHIKEVICH_TC, "LGN_OFF")

        # V1 populations
        self.v1_exc = IzhikevichPopulation(self.n_v1_exc, IZHIKEVICH_RS, "V1_Exc")
        self.v1_inh = IzhikevichPopulation(self.n_v1_inh, IZHIKEVICH_FS, "V1_Inh")

    def _init_synapses(self):
        """Initialize synaptic connections."""
        # RGC -> LGN: One-to-one connections (simple relay)
        self._init_rgc_to_lgn()

        # LGN -> V1: The key connectivity - patches to orientation ensembles
        self._init_lgn_to_v1()

        # V1 lateral connections (for competition)
        self._init_v1_lateral()

    def _init_rgc_to_lgn(self):
        """RGC to LGN: One-to-one relay connections."""
        # Simple one-to-one connectivity
        connectivity = cp.eye(self.n_rgc, dtype=bool)
        weights = cp.eye(self.n_rgc) * 15.0  # Strong feedforward drive
        delays = cp.ones((self.n_rgc, self.n_rgc), dtype=cp.int32) * 2  # 2ms delay

        # ON pathway
        self.syn_rgc_lgn_on = STDPSynapse(
            self.n_rgc, self.n_lgn, connectivity, weights, delays,
            w_max=20.0, w_min=5.0,
            A_plus=0.0, A_minus=0.0,  # No plasticity - fixed relay
            max_delay=5
        )

        # OFF pathway
        self.syn_rgc_lgn_off = STDPSynapse(
            self.n_rgc, self.n_lgn, connectivity, weights.copy(), delays.copy(),
            w_max=20.0, w_min=5.0,
            A_plus=0.0, A_minus=0.0,
            max_delay=5
        )

    def _init_lgn_to_v1(self):
        """
        LGN to V1: Each patch projects to multiple orientation-selective ensembles.

        This is the key connectivity implementing the hypothesis:
        - Each 4x4 LGN patch sends connections to all V1 neurons in its "orientation column"
        - Different initial weights and delays for each projection
        - STDP will select appropriate connections for each orientation
        """
        # Total LGN neurons (ON + OFF combined for V1 input)
        n_lgn_total = 2 * self.n_lgn

        # Initialize connectivity and weights
        connectivity = cp.zeros((n_lgn_total, self.n_v1_exc), dtype=bool)
        weights = cp.zeros((n_lgn_total, self.n_v1_exc))
        delays = cp.zeros((n_lgn_total, self.n_v1_exc), dtype=cp.int32)

        # For each patch
        for patch_y in range(self.n_patches_y):
            for patch_x in range(self.n_patches_x):
                patch_idx = patch_y * self.n_patches_x + patch_x

                # Get LGN neuron indices for this patch
                lgn_indices_on = []
                lgn_indices_off = []

                for dy in range(self.patch_size):
                    for dx in range(self.patch_size):
                        x = patch_x * self.patch_size + dx
                        y = patch_y * self.patch_size + dy

                        if x < self.retina_width and y < self.retina_height:
                            idx = y * self.retina_width + x
                            lgn_indices_on.append(idx)
                            lgn_indices_off.append(idx + self.n_lgn)  # OFF in second half

                lgn_indices = lgn_indices_on + lgn_indices_off

                # V1 neurons for this patch (the "orientation column")
                v1_start = patch_idx * self.n_orientations
                v1_end = v1_start + self.n_orientations

                # Connect each LGN neuron to each V1 orientation ensemble
                for lgn_idx in lgn_indices:
                    for v1_idx in range(v1_start, v1_end):
                        connectivity[lgn_idx, v1_idx] = True

                        # Initial weights: random with some structure
                        # Add slight bias based on position within patch
                        base_weight = 0.3 + 0.4 * np.random.random()
                        weights[lgn_idx, v1_idx] = base_weight

                        # Delays: vary by orientation preference
                        # This gives different temporal integration properties
                        orientation_idx = v1_idx - v1_start
                        base_delay = 3  # minimum delay
                        delay_range = self.max_delay - base_delay

                        # Different delays for different orientations
                        delay = base_delay + int((orientation_idx / self.n_orientations) * delay_range * 0.5)
                        delay += np.random.randint(0, 3)  # Add jitter
                        delays[lgn_idx, v1_idx] = min(delay, self.max_delay)

        # Convert to CuPy arrays
        connectivity = cp.asarray(connectivity)
        weights = cp.asarray(weights)
        delays = cp.asarray(delays)

        # Create STDP synapse
        self.syn_lgn_v1 = STDPSynapse(
            n_lgn_total, self.n_v1_exc,
            connectivity, weights, delays,
            w_max=self.stdp_params['w_max'],
            w_min=self.stdp_params['w_min'],
            tau_pre=self.stdp_params['tau_pre'],
            tau_post=self.stdp_params['tau_post'],
            A_plus=self.stdp_params['A_plus'],
            A_minus=self.stdp_params['A_minus'],
            max_delay=self.max_delay + 5
        )

        print(f"  LGN->V1 synapses: {int(cp.sum(connectivity))} connections")

    def _init_v1_lateral(self):
        """
        V1 lateral connections for competition.

        - Local excitation within same orientation preference (weak)
        - Lateral inhibition between different orientations (strong)
        """
        # V1 Exc -> V1 Inh (feedforward to inhibitory)
        # Each V1 exc drives local inhibitory neurons
        conn_exc_inh = cp.zeros((self.n_v1_exc, self.n_v1_inh), dtype=bool)
        weights_exc_inh = cp.zeros((self.n_v1_exc, self.n_v1_inh))

        inh_per_patch = self.n_v1_inh // self.n_patches

        for patch_idx in range(self.n_patches):
            exc_start = patch_idx * self.n_orientations
            exc_end = exc_start + self.n_orientations
            inh_start = patch_idx * inh_per_patch
            inh_end = inh_start + inh_per_patch

            # All excitatory neurons in patch drive all inhibitory neurons in patch
            for exc_idx in range(exc_start, exc_end):
                for inh_idx in range(inh_start, inh_end):
                    conn_exc_inh[exc_idx, inh_idx] = True
                    weights_exc_inh[exc_idx, inh_idx] = 0.5 + 0.3 * np.random.random()

        delays_exc_inh = cp.ones((self.n_v1_exc, self.n_v1_inh), dtype=cp.int32)

        self.syn_v1_exc_inh = STDPSynapse(
            self.n_v1_exc, self.n_v1_inh,
            conn_exc_inh, weights_exc_inh, delays_exc_inh,
            w_max=2.0, w_min=0.1,
            A_plus=0.001, A_minus=0.001,  # Weak plasticity
            max_delay=5
        )

        # V1 Inh -> V1 Exc (lateral inhibition)
        conn_inh_exc = cp.zeros((self.n_v1_inh, self.n_v1_exc), dtype=bool)
        weights_inh_exc = cp.zeros((self.n_v1_inh, self.n_v1_exc))

        for patch_idx in range(self.n_patches):
            exc_start = patch_idx * self.n_orientations
            exc_end = exc_start + self.n_orientations
            inh_start = patch_idx * inh_per_patch
            inh_end = inh_start + inh_per_patch

            # Inhibitory neurons suppress all excitatory neurons in patch
            for inh_idx in range(inh_start, inh_end):
                for exc_idx in range(exc_start, exc_end):
                    conn_inh_exc[inh_idx, exc_idx] = True
                    weights_inh_exc[inh_idx, exc_idx] = 1.5 + 0.5 * np.random.random()

        delays_inh_exc = cp.ones((self.n_v1_inh, self.n_v1_exc), dtype=cp.int32) * 2

        self.syn_v1_inh_exc = STDPSynapse(
            self.n_v1_inh, self.n_v1_exc,
            conn_inh_exc, weights_inh_exc, delays_inh_exc,
            w_max=5.0, w_min=0.5,
            A_plus=0.002, A_minus=0.002,  # Inhibitory STDP
            max_delay=5
        )

        print(f"  V1 Exc->Inh synapses: {int(cp.sum(conn_exc_inh))}")
        print(f"  V1 Inh->Exc synapses: {int(cp.sum(conn_inh_exc))}")

    def generate_poisson_spikes(self, rates: np.ndarray, dt: float = 1.0) -> cp.ndarray:
        """
        Generate Poisson spikes given firing rates.

        Args:
            rates: Firing rates in Hz for each neuron
            dt: Timestep in ms

        Returns:
            Boolean array indicating spikes
        """
        # Probability of spike in this timestep
        prob = rates * (dt / 1000.0)
        prob = np.clip(prob, 0, 1)

        # Generate spikes
        spikes = np.random.random(rates.shape) < prob
        return cp.asarray(spikes)

    def step(self, orientation: float, time_ms: float,
             learn: bool = True, dt: float = 1.0) -> Dict[str, cp.ndarray]:
        """
        Advance the network by one timestep.

        Args:
            orientation: Current stimulus orientation in degrees
            time_ms: Current time in ms
            learn: Whether to apply STDP learning
            dt: Timestep in ms

        Returns:
            Dictionary of spike arrays for each population
        """
        # Generate stimulus
        on_rates, off_rates = self.stim_gen.generate_spike_rates(
            orientation, time_ms, base_rate=5.0, max_rate=100.0
        )

        # Flatten to 1D
        on_rates = on_rates.flatten()
        off_rates = off_rates.flatten()

        # Generate RGC spikes directly from stimulus (Poisson)
        rgc_on_input = self.generate_poisson_spikes(on_rates, dt)
        rgc_off_input = self.generate_poisson_spikes(off_rates, dt)

        # RGC -> LGN
        lgn_on_current = self.syn_rgc_lgn_on.propagate(rgc_on_input.astype(cp.float64))
        lgn_off_current = self.syn_rgc_lgn_off.propagate(rgc_off_input.astype(cp.float64))

        # Add some baseline current to LGN
        lgn_on_current += 5.0
        lgn_off_current += 5.0

        # LGN dynamics
        lgn_on_fired = self.lgn_on.step(lgn_on_current, dt)
        lgn_off_fired = self.lgn_off.step(lgn_off_current, dt)

        # Combine LGN ON and OFF for V1 input
        lgn_combined = cp.concatenate([lgn_on_fired, lgn_off_fired])

        # LGN -> V1
        v1_exc_current = self.syn_lgn_v1.propagate(lgn_combined.astype(cp.float64))

        # V1 lateral: get inhibitory current
        v1_inh_current = self.syn_v1_exc_inh.propagate(self.v1_exc.fired.astype(cp.float64))
        v1_exc_lateral_inh = self.syn_v1_inh_exc.propagate(self.v1_inh.fired.astype(cp.float64))

        # Add baseline and combine currents
        v1_exc_total = v1_exc_current + 3.0 - v1_exc_lateral_inh * 0.5
        v1_inh_total = v1_inh_current + 2.0

        # V1 dynamics
        v1_exc_fired = self.v1_exc.step(v1_exc_total, dt)
        v1_inh_fired = self.v1_inh.step(v1_inh_total, dt)

        # STDP updates
        if learn:
            # LGN -> V1 STDP (the main learning)
            self.syn_lgn_v1.update_stdp(lgn_combined, v1_exc_fired, dt)

            # V1 lateral STDP (for competition dynamics)
            self.syn_v1_exc_inh.update_stdp(v1_exc_fired, v1_inh_fired, dt)
            self.syn_v1_inh_exc.update_stdp(v1_inh_fired, v1_exc_fired, dt)

        return {
            'rgc_on': rgc_on_input,
            'rgc_off': rgc_off_input,
            'lgn_on': lgn_on_fired,
            'lgn_off': lgn_off_fired,
            'v1_exc': v1_exc_fired,
            'v1_inh': v1_inh_fired
        }

    def run_trial(self, orientation: float, duration_ms: float = 500.0,
                  learn: bool = True, record: bool = True) -> Dict:
        """
        Run a single trial with a fixed orientation.

        Args:
            orientation: Stimulus orientation in degrees
            duration_ms: Trial duration in ms
            learn: Whether to apply STDP
            record: Whether to record spike times

        Returns:
            Trial statistics
        """
        n_steps = int(duration_ms)
        spike_counts = {
            'v1_exc': cp.zeros(self.n_v1_exc),
            'v1_inh': cp.zeros(self.n_v1_inh)
        }

        for t in range(n_steps):
            spikes = self.step(orientation, t, learn=learn)
            spike_counts['v1_exc'] += spikes['v1_exc'].astype(cp.float64)
            spike_counts['v1_inh'] += spikes['v1_inh'].astype(cp.float64)

            if record:
                for key in spikes:
                    self.spike_history[key].append(to_numpy(spikes[key]))

        return {
            'v1_exc_rates': to_numpy(spike_counts['v1_exc']) / (duration_ms / 1000.0),
            'v1_inh_rates': to_numpy(spike_counts['v1_inh']) / (duration_ms / 1000.0)
        }

    def train(self, n_epochs: int = 10,
              orientations: List[float] = None,
              trials_per_orientation: int = 5,
              trial_duration_ms: float = 500.0,
              visualize_every: int = 2,
              save_dir: str = "training_progress"):
        """
        Train the network on drifting gratings.

        Args:
            n_epochs: Number of training epochs
            orientations: List of orientations to train on (degrees)
            trials_per_orientation: Trials per orientation per epoch
            trial_duration_ms: Duration of each trial
            visualize_every: Visualize every N epochs
            save_dir: Directory to save visualizations
        """
        if orientations is None:
            orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

        os.makedirs(save_dir, exist_ok=True)

        print(f"\nStarting training:")
        print(f"  Epochs: {n_epochs}")
        print(f"  Orientations: {orientations}")
        print(f"  Trials per orientation: {trials_per_orientation}")
        print(f"  Trial duration: {trial_duration_ms} ms")

        # Track responses for each epoch
        epoch_responses = []

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            epoch_start = time.time()

            # Shuffle orientation order
            np.random.shuffle(orientations)

            epoch_data = {'orientations': [], 'responses': []}

            for ori in orientations:
                for trial in range(trials_per_orientation):
                    # Clear spike history
                    for key in self.spike_history:
                        self.spike_history[key] = []

                    # Reset neuron states between trials
                    self.rgc_on.reset()
                    self.rgc_off.reset()
                    self.lgn_on.reset()
                    self.lgn_off.reset()
                    self.v1_exc.reset()
                    self.v1_inh.reset()

                    # Run trial
                    results = self.run_trial(ori, trial_duration_ms, learn=True, record=False)

                    epoch_data['orientations'].append(ori)
                    epoch_data['responses'].append(results['v1_exc_rates'])

            epoch_responses.append(epoch_data)

            epoch_time = time.time() - epoch_start
            print(f"  Completed in {epoch_time:.1f}s")

            # Visualize progress
            if (epoch + 1) % visualize_every == 0 or epoch == 0 or epoch == n_epochs - 1:
                self.visualize_training_progress(epoch + 1, save_dir)
                self.visualize_tuning_curves(epoch + 1, orientations, save_dir)

        # Save final weight matrix
        self.weight_history.append(to_numpy(self.syn_lgn_v1.weights.copy()))

        print("\nTraining complete!")
        return epoch_responses

    def visualize_training_progress(self, epoch: int, save_dir: str):
        """Visualize LGN->V1 weight matrices as heatmaps."""
        weights = to_numpy(self.syn_lgn_v1.weights)

        # Create figure showing weights for first few patches
        n_show = min(4, self.n_patches)

        fig, axes = plt.subplots(n_show, self.n_orientations,
                                  figsize=(2 * self.n_orientations, 2 * n_show))
        fig.suptitle(f'LGN->V1 Weights - Epoch {epoch}\n(rows=patches, cols=orientation ensembles)')

        for patch_idx in range(n_show):
            v1_start = patch_idx * self.n_orientations

            # Get LGN indices for this patch
            patch_x = patch_idx % self.n_patches_x
            patch_y = patch_idx // self.n_patches_x

            for ori_idx in range(self.n_orientations):
                v1_idx = v1_start + ori_idx

                # Extract weights from all LGN neurons to this V1 neuron
                w_on = weights[:self.n_lgn, v1_idx].reshape(self.retina_height, self.retina_width)
                w_off = weights[self.n_lgn:, v1_idx].reshape(self.retina_height, self.retina_width)

                # Combined weight pattern
                w_combined = w_on - w_off  # ON - OFF gives edge-like structure

                ax = axes[patch_idx, ori_idx] if n_show > 1 else axes[ori_idx]
                im = ax.imshow(w_combined, cmap='RdBu_r', vmin=-1, vmax=1)

                # Highlight the patch region
                rect = plt.Rectangle(
                    (patch_x * self.patch_size - 0.5, patch_y * self.patch_size - 0.5),
                    self.patch_size, self.patch_size,
                    fill=False, color='green', linewidth=2
                )
                ax.add_patch(rect)

                ax.set_title(f'{ori_idx * 180 // self.n_orientations}°', fontsize=8)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/weights_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

        # Also plot total weight strength per V1 neuron
        fig, ax = plt.subplots(figsize=(12, 4))
        weight_sums = np.sum(weights, axis=0)

        # Color by orientation
        colors = plt.cm.hsv(np.linspace(0, 1, self.n_orientations))
        for i, c in enumerate(colors):
            indices = np.arange(i, self.n_v1_exc, self.n_orientations)
            ax.bar(indices, weight_sums[indices], color=c, width=0.8, alpha=0.7)

        ax.set_xlabel('V1 Neuron Index')
        ax.set_ylabel('Total Input Weight')
        ax.set_title(f'Total LGN->V1 Weight per Neuron - Epoch {epoch}')

        # Legend
        handles = [mpatches.Patch(color=colors[i], label=f'{i * 180 // self.n_orientations}°')
                   for i in range(self.n_orientations)]
        ax.legend(handles=handles, loc='upper right', ncol=4)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/weight_sums_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

        print(f"  Saved weight visualizations for epoch {epoch}")

    def visualize_tuning_curves(self, epoch: int, orientations: List[float],
                                 save_dir: str, test_trials: int = 3):
        """
        Measure and plot orientation tuning curves for V1 neurons.
        """
        # Test each orientation
        test_orientations = sorted(orientations)
        responses = {ori: [] for ori in test_orientations}

        for ori in test_orientations:
            for _ in range(test_trials):
                # Reset neurons
                self.rgc_on.reset()
                self.rgc_off.reset()
                self.lgn_on.reset()
                self.lgn_off.reset()
                self.v1_exc.reset()
                self.v1_inh.reset()

                # Reset synapse traces
                self.syn_lgn_v1.reset()

                # Run trial without learning
                results = self.run_trial(ori, 300.0, learn=False, record=False)
                responses[ori].append(results['v1_exc_rates'])

        # Average responses
        avg_responses = {ori: np.mean(responses[ori], axis=0) for ori in test_orientations}

        # Plot tuning curves for first few patches
        n_patches_show = min(4, self.n_patches)

        fig, axes = plt.subplots(n_patches_show, 2, figsize=(14, 3 * n_patches_show))
        fig.suptitle(f'Orientation Tuning - Epoch {epoch}')

        for patch_idx in range(n_patches_show):
            v1_start = patch_idx * self.n_orientations
            v1_end = v1_start + self.n_orientations

            # Left plot: tuning curves for each neuron in patch
            ax1 = axes[patch_idx, 0] if n_patches_show > 1 else axes[0]

            for neuron_idx in range(v1_start, v1_end):
                neuron_responses = [avg_responses[ori][neuron_idx] for ori in test_orientations]
                ori_label = (neuron_idx - v1_start) * 180 // self.n_orientations
                ax1.plot(test_orientations, neuron_responses, 'o-',
                        label=f'N{neuron_idx - v1_start} ({ori_label}°)')

            ax1.set_xlabel('Stimulus Orientation (°)')
            ax1.set_ylabel('Firing Rate (Hz)')
            ax1.set_title(f'Patch {patch_idx}: Tuning Curves')
            ax1.legend(fontsize=6, ncol=2)
            ax1.set_xticks(test_orientations)

            # Right plot: preferred orientation for each neuron
            ax2 = axes[patch_idx, 1] if n_patches_show > 1 else axes[1]

            preferred_oris = []
            selectivity_indices = []

            for neuron_idx in range(v1_start, v1_end):
                neuron_responses = np.array([avg_responses[ori][neuron_idx] for ori in test_orientations])

                if np.max(neuron_responses) > 0:
                    # Preferred orientation
                    pref_idx = np.argmax(neuron_responses)
                    preferred_oris.append(test_orientations[pref_idx])

                    # Orientation Selectivity Index (OSI)
                    # OSI = (R_pref - R_orth) / (R_pref + R_orth)
                    orth_idx = (pref_idx + len(test_orientations) // 2) % len(test_orientations)
                    r_pref = neuron_responses[pref_idx]
                    r_orth = neuron_responses[orth_idx]
                    osi = (r_pref - r_orth) / (r_pref + r_orth + 1e-6)
                    selectivity_indices.append(osi)
                else:
                    preferred_oris.append(0)
                    selectivity_indices.append(0)

            x = np.arange(self.n_orientations)
            expected_oris = [i * 180 // self.n_orientations for i in range(self.n_orientations)]

            bars = ax2.bar(x, preferred_oris, alpha=0.7)
            ax2.plot(x, expected_oris, 'r--', linewidth=2, label='Expected')
            ax2.set_xlabel('Neuron Index in Patch')
            ax2.set_ylabel('Preferred Orientation (°)')
            ax2.set_title(f'Patch {patch_idx}: Preferred Orientations')
            ax2.legend()

            # Color bars by OSI
            for i, (bar, osi) in enumerate(zip(bars, selectivity_indices)):
                bar.set_color(plt.cm.viridis(osi))

        plt.tight_layout()
        plt.savefig(f'{save_dir}/tuning_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

        # Plot OSI histogram
        all_osis = []
        for patch_idx in range(self.n_patches):
            v1_start = patch_idx * self.n_orientations
            v1_end = v1_start + self.n_orientations

            for neuron_idx in range(v1_start, v1_end):
                neuron_responses = np.array([avg_responses[ori][neuron_idx] for ori in test_orientations])
                if np.max(neuron_responses) > 0:
                    pref_idx = np.argmax(neuron_responses)
                    orth_idx = (pref_idx + len(test_orientations) // 2) % len(test_orientations)
                    r_pref = neuron_responses[pref_idx]
                    r_orth = neuron_responses[orth_idx]
                    osi = (r_pref - r_orth) / (r_pref + r_orth + 1e-6)
                    all_osis.append(osi)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(all_osis, bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(all_osis), color='red', linestyle='--',
                   label=f'Mean OSI: {np.mean(all_osis):.3f}')
        ax.set_xlabel('Orientation Selectivity Index (OSI)')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of OSI - Epoch {epoch}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'{save_dir}/osi_distribution_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

        print(f"  Mean OSI: {np.mean(all_osis):.3f}")

    def plot_receptive_fields(self, save_path: str = "receptive_fields.png"):
        """
        Visualize the effective receptive fields of V1 neurons
        based on their LGN input weights.
        """
        weights = to_numpy(self.syn_lgn_v1.weights)

        n_patches_show = min(4, self.n_patches)

        fig, axes = plt.subplots(n_patches_show, self.n_orientations,
                                  figsize=(2 * self.n_orientations, 2 * n_patches_show))
        fig.suptitle('Effective Receptive Fields (ON-OFF weight patterns)')

        for patch_idx in range(n_patches_show):
            v1_start = patch_idx * self.n_orientations

            patch_x = patch_idx % self.n_patches_x
            patch_y = patch_idx // self.n_patches_x

            # Extract weights just for this patch region
            x_start = patch_x * self.patch_size
            x_end = x_start + self.patch_size
            y_start = patch_y * self.patch_size
            y_end = y_start + self.patch_size

            for ori_idx in range(self.n_orientations):
                v1_idx = v1_start + ori_idx

                w_on = weights[:self.n_lgn, v1_idx].reshape(self.retina_height, self.retina_width)
                w_off = weights[self.n_lgn:, v1_idx].reshape(self.retina_height, self.retina_width)

                # RF in patch region
                rf = w_on[y_start:y_end, x_start:x_end] - w_off[y_start:y_end, x_start:x_end]

                ax = axes[patch_idx, ori_idx] if n_patches_show > 1 else axes[ori_idx]
                im = ax.imshow(rf, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
                ax.set_title(f'{ori_idx * 180 // self.n_orientations}°', fontsize=8)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved receptive fields to {save_path}")


def main():
    """Main training loop with comprehensive visualization."""
    print("=" * 60)
    print("Orientation Selectivity Network Training")
    print("=" * 60)

    # Create network
    network = OrientationSelectivityNetwork(
        retina_size=(16, 16),     # 16x16 retina/LGN
        lgn_patch_size=4,          # 4x4 patches
        n_orientations=8,          # 8 orientation ensembles per patch
        max_delay=15,
        stdp_params={
            'tau_pre': 20.0,
            'tau_post': 20.0,
            'A_plus': 0.008,       # Learning rate for potentiation
            'A_minus': 0.01,       # Slightly stronger depression
            'w_max': 1.5,
            'w_min': 0.01
        }
    )

    # Training orientations (8 equally spaced)
    orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

    # Train the network
    results = network.train(
        n_epochs=15,
        orientations=orientations,
        trials_per_orientation=8,
        trial_duration_ms=400.0,
        visualize_every=3,
        save_dir="training_progress"
    )

    # Final visualizations
    print("\nGenerating final visualizations...")
    network.plot_receptive_fields("training_progress/final_receptive_fields.png")

    print("\n" + "=" * 60)
    print("Training complete! Check 'training_progress' folder for results.")
    print("=" * 60)

    return network, results


if __name__ == "__main__":
    network, results = main()
