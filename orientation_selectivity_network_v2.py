"""
Spiking Neural Network for Orientation Selectivity Emergence - Version 2
RGC (ON/OFF) -> LGN -> V1 Layer 4

Improved version with:
- Better structured initial weights with orientation bias
- Stronger competitive lateral inhibition
- Weight-dependent STDP (soft bounds)
- Enhanced visualization

Author: Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
from typing import Tuple, Dict, List, Optional
import time
import os

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy available - using GPU acceleration")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU (NumPy)")


def to_numpy(arr):
    """Convert array to numpy."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.array(arr)


# ==============================================================================
# IZHIKEVICH NEURON PARAMETERS
# Based on Izhikevich (2003) "Simple Model of Spiking Neurons"
# ==============================================================================

IZHIKEVICH_RS = {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}  # Regular Spiking
IZHIKEVICH_FS = {'a': 0.1, 'b': 0.2, 'c': -65.0, 'd': 2.0}   # Fast Spiking
IZHIKEVICH_TC = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05} # Thalamic
IZHIKEVICH_RGC = {'a': 0.03, 'b': 0.25, 'c': -65.0, 'd': 4.0} # Retinal


class IzhikevichPopulation:
    """Population of Izhikevich neurons."""

    def __init__(self, n_neurons: int, params: dict, name: str = ""):
        self.n = n_neurons
        self.name = name
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']

        self.v = cp.ones(n_neurons) * -65.0
        self.u = cp.ones(n_neurons) * self.b * (-65.0)
        self.fired = cp.zeros(n_neurons, dtype=bool)

    def step(self, I: cp.ndarray, dt: float = 1.0) -> cp.ndarray:
        """Advance by one timestep using two half-steps for stability."""
        for _ in range(2):
            dv = (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + I) * (dt/2)
            self.v = self.v + dv
            du = self.a * (self.b * self.v - self.u) * (dt/2)
            self.u = self.u + du

        self.fired = self.v >= 30.0
        self.v = cp.where(self.fired, self.c, self.v)
        self.u = cp.where(self.fired, self.u + self.d, self.u)
        return self.fired

    def reset(self):
        self.v = cp.ones(self.n) * -65.0
        self.u = cp.ones(self.n) * self.b * (-65.0)
        self.fired = cp.zeros(self.n, dtype=bool)


class STDPSynapse:
    """
    STDP synapse with weight-dependent learning (soft bounds).

    Potentiation: dw = A_plus * (w_max - w) * exp(-|dt|/tau)
    Depression: dw = -A_minus * w * exp(-|dt|/tau)
    """

    def __init__(self, n_pre: int, n_post: int,
                 connectivity: cp.ndarray,
                 w_init: cp.ndarray,
                 delays: cp.ndarray,
                 w_max: float = 1.0,
                 w_min: float = 0.0,
                 tau_pre: float = 20.0,
                 tau_post: float = 20.0,
                 A_plus: float = 0.01,
                 A_minus: float = 0.012,
                 max_delay: int = 20):

        self.n_pre = n_pre
        self.n_post = n_post
        self.connectivity = connectivity.astype(bool)
        self.weights = w_init * connectivity
        self.delays = delays.astype(cp.int32)
        self.max_delay = max_delay

        self.w_max = w_max
        self.w_min = w_min
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_plus = A_plus
        self.A_minus = A_minus

        self.trace_pre = cp.zeros(n_pre)
        self.trace_post = cp.zeros(n_post)

        self.spike_buffer = cp.zeros((max_delay + 1, n_pre), dtype=bool)
        self.buffer_idx = 0

    def propagate(self, pre_fired: cp.ndarray) -> cp.ndarray:
        """Propagate spikes with delays."""
        self.spike_buffer[self.buffer_idx] = pre_fired

        post_current = cp.zeros(self.n_post)
        unique_delays = cp.unique(self.delays[self.connectivity])

        for delay in unique_delays:
            delay_int = int(delay)
            buffer_read_idx = (self.buffer_idx - delay_int) % (self.max_delay + 1)
            delayed_spikes = self.spike_buffer[buffer_read_idx]

            delay_mask = (self.delays == delay_int) & self.connectivity
            spike_matrix = cp.outer(delayed_spikes.astype(cp.float64), cp.ones(self.n_post))
            contribution = cp.sum(self.weights * delay_mask * spike_matrix, axis=0)
            post_current += contribution

        self.buffer_idx = (self.buffer_idx + 1) % (self.max_delay + 1)
        return post_current

    def update_stdp(self, pre_fired: cp.ndarray, post_fired: cp.ndarray, dt: float = 1.0):
        """Update weights with weight-dependent STDP."""
        decay_pre = cp.exp(-dt / self.tau_pre)
        decay_post = cp.exp(-dt / self.tau_post)

        self.trace_pre = self.trace_pre * decay_pre
        self.trace_post = self.trace_post * decay_post

        self.trace_pre = cp.where(pre_fired, self.trace_pre + 1.0, self.trace_pre)
        self.trace_post = cp.where(post_fired, self.trace_post + 1.0, self.trace_post)

        # Weight-dependent STDP (soft bounds)
        if cp.any(post_fired):
            # Potentiation: proportional to (w_max - w)
            potentiation_factor = (self.w_max - self.weights) / self.w_max
            dw_plus = self.A_plus * potentiation_factor * cp.outer(
                self.trace_pre, post_fired.astype(cp.float64))
            self.weights = self.weights + dw_plus * self.connectivity

        if cp.any(pre_fired):
            # Depression: proportional to w
            depression_factor = self.weights / self.w_max
            dw_minus = self.A_minus * depression_factor * cp.outer(
                pre_fired.astype(cp.float64), self.trace_post)
            self.weights = self.weights - dw_minus * self.connectivity

        self.weights = cp.clip(self.weights, self.w_min, self.w_max)

    def reset(self):
        self.trace_pre = cp.zeros(self.n_pre)
        self.trace_post = cp.zeros(self.n_post)
        self.spike_buffer = cp.zeros((self.max_delay + 1, self.n_pre), dtype=bool)
        self.buffer_idx = 0


class DriftingGratingGenerator:
    """Generates drifting sinusoidal grating stimuli."""

    def __init__(self, width: int, height: int,
                 spatial_freq: float = 0.15,
                 temporal_freq: float = 2.0,
                 contrast: float = 1.0):
        self.width = width
        self.height = height
        self.spatial_freq = spatial_freq
        self.temporal_freq = temporal_freq
        self.contrast = contrast

        x = np.arange(width)
        y = np.arange(height)
        self.X, self.Y = np.meshgrid(x, y)

    def generate(self, orientation: float, time_ms: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ON and OFF responses for drifting grating."""
        theta = np.radians(orientation)
        phase = 2 * np.pi * self.temporal_freq * (time_ms / 1000.0)

        spatial_phase = 2 * np.pi * self.spatial_freq * (
            self.X * np.cos(theta) + self.Y * np.sin(theta)
        )

        grating = self.contrast * np.sin(spatial_phase + phase)

        on_response = np.maximum(0, grating)
        off_response = np.maximum(0, -grating)

        return on_response, off_response

    def generate_spike_rates(self, orientation: float, time_ms: float,
                            base_rate: float = 10.0,
                            max_rate: float = 150.0) -> Tuple[np.ndarray, np.ndarray]:
        """Convert grating to spike rates."""
        on_resp, off_resp = self.generate(orientation, time_ms)

        on_rates = base_rate + (max_rate - base_rate) * on_resp
        off_rates = base_rate + (max_rate - base_rate) * off_resp

        return on_rates, off_rates


def create_oriented_weight_template(patch_size: int, orientation: float,
                                     width: float = 1.0) -> np.ndarray:
    """
    Create a Gabor-like weight template for a given orientation.

    This provides a slight initial bias toward certain orientations.
    """
    x = np.arange(patch_size) - (patch_size - 1) / 2
    y = np.arange(patch_size) - (patch_size - 1) / 2
    X, Y = np.meshgrid(x, y)

    theta = np.radians(orientation)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    # Simple Gabor-like pattern
    freq = 0.5  # spatial frequency within patch
    envelope = np.exp(-(X_rot**2 + Y_rot**2) / (2 * width**2))
    pattern = envelope * np.cos(2 * np.pi * freq * Y_rot)

    # Normalize to [0, 1]
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)

    return pattern


class OrientationSelectivityNetwork:
    """
    Complete RGC -> LGN -> V1 network with orientation selectivity emergence.
    """

    def __init__(self,
                 retina_size: Tuple[int, int] = (16, 16),
                 lgn_patch_size: int = 4,
                 n_orientations: int = 8,
                 max_delay: int = 15,
                 initial_bias_strength: float = 0.3,
                 stdp_params: Optional[dict] = None):

        self.retina_width, self.retina_height = retina_size
        self.patch_size = lgn_patch_size
        self.n_orientations = n_orientations
        self.max_delay = max_delay
        self.initial_bias_strength = initial_bias_strength

        self.n_patches_x = self.retina_width // lgn_patch_size
        self.n_patches_y = self.retina_height // lgn_patch_size
        self.n_patches = self.n_patches_x * self.n_patches_y

        self.n_rgc = self.retina_width * self.retina_height
        self.n_lgn = self.n_rgc
        self.n_v1_exc = self.n_patches * n_orientations
        self.n_v1_inh = self.n_patches * (n_orientations // 2)

        print(f"Network Architecture:")
        print(f"  Retina: {retina_size[0]}x{retina_size[1]}")
        print(f"  LGN patches: {self.n_patches_x}x{self.n_patches_y} = {self.n_patches}")
        print(f"  V1 Excitatory: {self.n_v1_exc} ({n_orientations} per patch)")
        print(f"  V1 Inhibitory: {self.n_v1_inh}")

        self.stdp_params = stdp_params or {
            'tau_pre': 20.0,
            'tau_post': 20.0,
            'A_plus': 0.015,
            'A_minus': 0.02,
            'w_max': 2.0,
            'w_min': 0.01
        }

        self._init_neurons()
        self._init_synapses()

        self.stim_gen = DriftingGratingGenerator(
            self.retina_width, self.retina_height,
            spatial_freq=0.2,
            temporal_freq=3.0
        )

        self.spike_history = {}
        self.weight_history = []

    def _init_neurons(self):
        self.rgc_on = IzhikevichPopulation(self.n_rgc, IZHIKEVICH_RGC, "RGC_ON")
        self.rgc_off = IzhikevichPopulation(self.n_rgc, IZHIKEVICH_RGC, "RGC_OFF")
        self.lgn_on = IzhikevichPopulation(self.n_lgn, IZHIKEVICH_TC, "LGN_ON")
        self.lgn_off = IzhikevichPopulation(self.n_lgn, IZHIKEVICH_TC, "LGN_OFF")
        self.v1_exc = IzhikevichPopulation(self.n_v1_exc, IZHIKEVICH_RS, "V1_Exc")
        self.v1_inh = IzhikevichPopulation(self.n_v1_inh, IZHIKEVICH_FS, "V1_Inh")

    def _init_synapses(self):
        self._init_rgc_to_lgn()
        self._init_lgn_to_v1()
        self._init_v1_lateral()

    def _init_rgc_to_lgn(self):
        """RGC to LGN: One-to-one relay."""
        connectivity = cp.eye(self.n_rgc, dtype=bool)
        weights = cp.eye(self.n_rgc) * 20.0
        delays = cp.ones((self.n_rgc, self.n_rgc), dtype=cp.int32) * 2

        self.syn_rgc_lgn_on = STDPSynapse(
            self.n_rgc, self.n_lgn, connectivity, weights, delays,
            w_max=25.0, w_min=10.0, A_plus=0.0, A_minus=0.0, max_delay=5
        )

        self.syn_rgc_lgn_off = STDPSynapse(
            self.n_rgc, self.n_lgn, connectivity, weights.copy(), delays.copy(),
            w_max=25.0, w_min=10.0, A_plus=0.0, A_minus=0.0, max_delay=5
        )

    def _init_lgn_to_v1(self):
        """
        LGN to V1 with orientation-biased initial weights.

        Each 4x4 patch projects to N orientation ensembles.
        Initial weights have a slight Gabor-like bias.
        """
        n_lgn_total = 2 * self.n_lgn

        connectivity = cp.zeros((n_lgn_total, self.n_v1_exc), dtype=bool)
        weights = cp.zeros((n_lgn_total, self.n_v1_exc))
        delays = cp.zeros((n_lgn_total, self.n_v1_exc), dtype=cp.int32)

        # Create orientation templates
        ori_angles = np.linspace(0, 180, self.n_orientations, endpoint=False)

        for patch_y in range(self.n_patches_y):
            for patch_x in range(self.n_patches_x):
                patch_idx = patch_y * self.n_patches_x + patch_x

                # Get LGN indices for patch
                lgn_indices_on = []
                lgn_indices_off = []
                local_coords = []

                for dy in range(self.patch_size):
                    for dx in range(self.patch_size):
                        x = patch_x * self.patch_size + dx
                        y = patch_y * self.patch_size + dy

                        if x < self.retina_width and y < self.retina_height:
                            idx = y * self.retina_width + x
                            lgn_indices_on.append(idx)
                            lgn_indices_off.append(idx + self.n_lgn)
                            local_coords.append((dy, dx))

                v1_start = patch_idx * self.n_orientations

                for ori_idx in range(self.n_orientations):
                    v1_idx = v1_start + ori_idx
                    ori_angle = ori_angles[ori_idx]

                    # Create weight template
                    template = create_oriented_weight_template(
                        self.patch_size, ori_angle, width=1.5
                    )

                    for i, (lgn_on, lgn_off) in enumerate(zip(lgn_indices_on, lgn_indices_off)):
                        dy, dx = local_coords[i]

                        connectivity[lgn_on, v1_idx] = True
                        connectivity[lgn_off, v1_idx] = True

                        # Base weight with orientation bias
                        base_weight = 0.3 + 0.2 * np.random.random()
                        bias = self.initial_bias_strength * template[dy, dx]

                        # ON channel gets positive lobe, OFF gets negative lobe
                        weights[lgn_on, v1_idx] = base_weight + bias
                        weights[lgn_off, v1_idx] = base_weight + (self.initial_bias_strength - bias)

                        # Variable delays
                        base_delay = 3
                        delay_variation = int(np.random.random() * 5)
                        delays[lgn_on, v1_idx] = base_delay + delay_variation
                        delays[lgn_off, v1_idx] = base_delay + delay_variation + 1

        connectivity = cp.asarray(connectivity)
        weights = cp.asarray(weights)
        delays = cp.asarray(delays)

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

        print(f"  LGN->V1 synapses: {int(cp.sum(connectivity))}")

    def _init_v1_lateral(self):
        """Strong lateral inhibition for competition."""
        inh_per_patch = self.n_v1_inh // self.n_patches

        # V1 Exc -> Inh
        conn_exc_inh = cp.zeros((self.n_v1_exc, self.n_v1_inh), dtype=bool)
        weights_exc_inh = cp.zeros((self.n_v1_exc, self.n_v1_inh))

        for patch_idx in range(self.n_patches):
            exc_start = patch_idx * self.n_orientations
            exc_end = exc_start + self.n_orientations
            inh_start = patch_idx * inh_per_patch
            inh_end = inh_start + inh_per_patch

            for exc_idx in range(exc_start, exc_end):
                for inh_idx in range(inh_start, inh_end):
                    conn_exc_inh[exc_idx, inh_idx] = True
                    weights_exc_inh[exc_idx, inh_idx] = 1.0 + 0.5 * np.random.random()

        delays_exc_inh = cp.ones((self.n_v1_exc, self.n_v1_inh), dtype=cp.int32)

        self.syn_v1_exc_inh = STDPSynapse(
            self.n_v1_exc, self.n_v1_inh,
            conn_exc_inh, weights_exc_inh, delays_exc_inh,
            w_max=3.0, w_min=0.2, A_plus=0.002, A_minus=0.002, max_delay=5
        )

        # V1 Inh -> Exc (strong lateral inhibition)
        conn_inh_exc = cp.zeros((self.n_v1_inh, self.n_v1_exc), dtype=bool)
        weights_inh_exc = cp.zeros((self.n_v1_inh, self.n_v1_exc))

        for patch_idx in range(self.n_patches):
            exc_start = patch_idx * self.n_orientations
            exc_end = exc_start + self.n_orientations
            inh_start = patch_idx * inh_per_patch
            inh_end = inh_start + inh_per_patch

            for inh_idx in range(inh_start, inh_end):
                for exc_idx in range(exc_start, exc_end):
                    conn_inh_exc[inh_idx, exc_idx] = True
                    weights_inh_exc[inh_idx, exc_idx] = 2.5 + 1.0 * np.random.random()

        delays_inh_exc = cp.ones((self.n_v1_inh, self.n_v1_exc), dtype=cp.int32) * 2

        self.syn_v1_inh_exc = STDPSynapse(
            self.n_v1_inh, self.n_v1_exc,
            conn_inh_exc, weights_inh_exc, delays_inh_exc,
            w_max=8.0, w_min=1.0, A_plus=0.003, A_minus=0.003, max_delay=5
        )

        print(f"  V1 lateral connections initialized")

    def generate_poisson_spikes(self, rates: np.ndarray, dt: float = 1.0) -> cp.ndarray:
        prob = rates * (dt / 1000.0)
        prob = np.clip(prob, 0, 1)
        spikes = np.random.random(rates.shape) < prob
        return cp.asarray(spikes)

    def step(self, orientation: float, time_ms: float,
             learn: bool = True, dt: float = 1.0) -> Dict[str, cp.ndarray]:
        """Advance network by one timestep."""

        on_rates, off_rates = self.stim_gen.generate_spike_rates(
            orientation, time_ms, base_rate=10.0, max_rate=150.0
        )

        on_rates = on_rates.flatten()
        off_rates = off_rates.flatten()

        rgc_on_spikes = self.generate_poisson_spikes(on_rates, dt)
        rgc_off_spikes = self.generate_poisson_spikes(off_rates, dt)

        lgn_on_current = self.syn_rgc_lgn_on.propagate(rgc_on_spikes.astype(cp.float64))
        lgn_off_current = self.syn_rgc_lgn_off.propagate(rgc_off_spikes.astype(cp.float64))

        lgn_on_current += 8.0
        lgn_off_current += 8.0

        lgn_on_fired = self.lgn_on.step(lgn_on_current, dt)
        lgn_off_fired = self.lgn_off.step(lgn_off_current, dt)

        lgn_combined = cp.concatenate([lgn_on_fired, lgn_off_fired])

        v1_exc_current = self.syn_lgn_v1.propagate(lgn_combined.astype(cp.float64))
        v1_inh_current = self.syn_v1_exc_inh.propagate(self.v1_exc.fired.astype(cp.float64))
        v1_exc_lateral_inh = self.syn_v1_inh_exc.propagate(self.v1_inh.fired.astype(cp.float64))

        # Normalize feedforward input by patch
        v1_exc_total = v1_exc_current + 5.0 - v1_exc_lateral_inh * 0.8
        v1_inh_total = v1_inh_current + 3.0

        v1_exc_fired = self.v1_exc.step(v1_exc_total, dt)
        v1_inh_fired = self.v1_inh.step(v1_inh_total, dt)

        if learn:
            self.syn_lgn_v1.update_stdp(lgn_combined, v1_exc_fired, dt)
            self.syn_v1_exc_inh.update_stdp(v1_exc_fired, v1_inh_fired, dt)
            self.syn_v1_inh_exc.update_stdp(v1_inh_fired, v1_exc_fired, dt)

        return {
            'rgc_on': rgc_on_spikes, 'rgc_off': rgc_off_spikes,
            'lgn_on': lgn_on_fired, 'lgn_off': lgn_off_fired,
            'v1_exc': v1_exc_fired, 'v1_inh': v1_inh_fired
        }

    def run_trial(self, orientation: float, duration_ms: float = 500.0,
                  learn: bool = True) -> Dict:
        n_steps = int(duration_ms)
        spike_counts = cp.zeros(self.n_v1_exc)

        for t in range(n_steps):
            spikes = self.step(orientation, t, learn=learn)
            spike_counts += spikes['v1_exc'].astype(cp.float64)

        return {
            'v1_exc_rates': to_numpy(spike_counts) / (duration_ms / 1000.0)
        }

    def measure_orientation_tuning(self, orientations: List[float],
                                    test_trials: int = 3,
                                    trial_duration: float = 300.0) -> Dict:
        """Measure orientation tuning curves for all V1 neurons."""
        responses = {ori: [] for ori in orientations}

        for ori in orientations:
            for _ in range(test_trials):
                self._reset_all()
                results = self.run_trial(ori, trial_duration, learn=False)
                responses[ori].append(results['v1_exc_rates'])

        avg_responses = {ori: np.mean(responses[ori], axis=0) for ori in orientations}

        # Compute tuning metrics
        tuning_data = []
        for neuron_idx in range(self.n_v1_exc):
            neuron_resp = np.array([avg_responses[ori][neuron_idx] for ori in sorted(orientations)])

            if np.max(neuron_resp) > 0:
                pref_idx = np.argmax(neuron_resp)
                pref_ori = sorted(orientations)[pref_idx]

                # OSI calculation
                orth_idx = (pref_idx + len(orientations) // 2) % len(orientations)
                r_pref = neuron_resp[pref_idx]
                r_orth = neuron_resp[orth_idx]
                osi = (r_pref - r_orth) / (r_pref + r_orth + 1e-6)

                # Circular variance
                theta_rad = np.radians(np.array(sorted(orientations)) * 2)
                r_norm = neuron_resp / (np.sum(neuron_resp) + 1e-6)
                cv = 1 - np.abs(np.sum(r_norm * np.exp(1j * theta_rad)))
            else:
                pref_ori = 0
                osi = 0
                cv = 1

            tuning_data.append({
                'neuron_idx': neuron_idx,
                'preferred_ori': pref_ori,
                'osi': osi,
                'circular_variance': cv,
                'responses': neuron_resp
            })

        return {
            'avg_responses': avg_responses,
            'tuning_data': tuning_data,
            'mean_osi': np.mean([d['osi'] for d in tuning_data]),
            'mean_cv': np.mean([d['circular_variance'] for d in tuning_data])
        }

    def _reset_all(self):
        """Reset all neuron and synapse states."""
        for pop in [self.rgc_on, self.rgc_off, self.lgn_on, self.lgn_off,
                    self.v1_exc, self.v1_inh]:
            pop.reset()
        for syn in [self.syn_rgc_lgn_on, self.syn_rgc_lgn_off, self.syn_lgn_v1,
                    self.syn_v1_exc_inh, self.syn_v1_inh_exc]:
            syn.reset()

    def train(self, n_epochs: int = 20,
              orientations: List[float] = None,
              trials_per_orientation: int = 10,
              trial_duration_ms: float = 400.0,
              visualize_every: int = 5,
              save_dir: str = "training_progress"):
        """Train the network."""

        if orientations is None:
            orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

        os.makedirs(save_dir, exist_ok=True)

        print(f"\nTraining Configuration:")
        print(f"  Epochs: {n_epochs}")
        print(f"  Orientations: {orientations}")
        print(f"  Trials/orientation: {trials_per_orientation}")
        print(f"  Trial duration: {trial_duration_ms} ms")

        history = {'epoch': [], 'mean_osi': [], 'mean_cv': []}

        # Initial measurement
        print("\nMeasuring initial tuning...")
        initial_tuning = self.measure_orientation_tuning(orientations)
        print(f"  Initial Mean OSI: {initial_tuning['mean_osi']:.4f}")
        history['epoch'].append(0)
        history['mean_osi'].append(initial_tuning['mean_osi'])
        history['mean_cv'].append(initial_tuning['mean_cv'])

        self.visualize_all(0, orientations, save_dir, initial_tuning)

        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            epoch_start = time.time()

            ori_order = np.random.permutation(orientations)

            for ori in ori_order:
                for trial in range(trials_per_orientation):
                    self._reset_all()
                    self.run_trial(ori, trial_duration_ms, learn=True)

            epoch_time = time.time() - epoch_start
            print(f"  Training time: {epoch_time:.1f}s")

            # Measure tuning
            tuning = self.measure_orientation_tuning(orientations)
            print(f"  Mean OSI: {tuning['mean_osi']:.4f}")

            history['epoch'].append(epoch)
            history['mean_osi'].append(tuning['mean_osi'])
            history['mean_cv'].append(tuning['mean_cv'])

            if epoch % visualize_every == 0 or epoch == n_epochs:
                self.visualize_all(epoch, orientations, save_dir, tuning)

        # Plot training history
        self._plot_training_history(history, save_dir)

        print(f"\nTraining complete!")
        print(f"  Final Mean OSI: {history['mean_osi'][-1]:.4f}")
        print(f"  Results saved to: {save_dir}")

        return history

    def visualize_all(self, epoch: int, orientations: List[float],
                      save_dir: str, tuning_data: Dict):
        """Generate comprehensive visualizations."""

        # 1. Weight heatmaps
        self._plot_weight_patterns(epoch, save_dir)

        # 2. Tuning curves
        self._plot_tuning_curves(epoch, orientations, tuning_data, save_dir)

        # 3. Orientation map
        self._plot_orientation_map(epoch, tuning_data, save_dir)

        # 4. OSI histogram
        self._plot_osi_histogram(epoch, tuning_data, save_dir)

    def _plot_weight_patterns(self, epoch: int, save_dir: str):
        """Plot weight patterns for each V1 neuron."""
        weights = to_numpy(self.syn_lgn_v1.weights)

        n_show = min(4, self.n_patches)

        fig, axes = plt.subplots(n_show, self.n_orientations,
                                  figsize=(2 * self.n_orientations, 2 * n_show))
        fig.suptitle(f'LGN->V1 Weight Patterns (ON - OFF) - Epoch {epoch}')

        for patch_idx in range(n_show):
            v1_start = patch_idx * self.n_orientations
            patch_x = patch_idx % self.n_patches_x
            patch_y = patch_idx // self.n_patches_x

            x_start = patch_x * self.patch_size
            x_end = x_start + self.patch_size
            y_start = patch_y * self.patch_size
            y_end = y_start + self.patch_size

            for ori_idx in range(self.n_orientations):
                v1_idx = v1_start + ori_idx

                w_on = weights[:self.n_lgn, v1_idx].reshape(
                    self.retina_height, self.retina_width)
                w_off = weights[self.n_lgn:, v1_idx].reshape(
                    self.retina_height, self.retina_width)

                rf = w_on[y_start:y_end, x_start:x_end] - w_off[y_start:y_end, x_start:x_end]

                ax = axes[patch_idx, ori_idx] if n_show > 1 else axes[ori_idx]
                im = ax.imshow(rf, cmap='RdBu_r', vmin=-1, vmax=1)
                expected_ori = ori_idx * 180 // self.n_orientations
                ax.set_title(f'{expected_ori}°', fontsize=8)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/weights_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

    def _plot_tuning_curves(self, epoch: int, orientations: List[float],
                            tuning_data: Dict, save_dir: str):
        """Plot orientation tuning curves."""
        n_show = min(4, self.n_patches)

        fig, axes = plt.subplots(n_show, 2, figsize=(12, 3 * n_show))
        fig.suptitle(f'Orientation Tuning - Epoch {epoch}')

        sorted_oris = sorted(orientations)
        avg_responses = tuning_data['avg_responses']

        for patch_idx in range(n_show):
            v1_start = patch_idx * self.n_orientations
            v1_end = v1_start + self.n_orientations

            # Tuning curves
            ax1 = axes[patch_idx, 0] if n_show > 1 else axes[0]

            colors = plt.cm.hsv(np.linspace(0, 1, self.n_orientations))
            for i, neuron_idx in enumerate(range(v1_start, v1_end)):
                responses = [avg_responses[ori][neuron_idx] for ori in sorted_oris]
                expected_ori = i * 180 // self.n_orientations
                ax1.plot(sorted_oris, responses, 'o-', color=colors[i],
                        label=f'N{i} ({expected_ori}°)', alpha=0.7)

            ax1.set_xlabel('Stimulus Orientation (°)')
            ax1.set_ylabel('Firing Rate (Hz)')
            ax1.set_title(f'Patch {patch_idx}: Tuning Curves')
            ax1.legend(fontsize=6, ncol=2)

            # Preferred orientation comparison
            ax2 = axes[patch_idx, 1] if n_show > 1 else axes[1]

            expected_oris = [i * 180 // self.n_orientations for i in range(self.n_orientations)]
            actual_oris = [tuning_data['tuning_data'][v1_start + i]['preferred_ori']
                          for i in range(self.n_orientations)]
            osis = [tuning_data['tuning_data'][v1_start + i]['osi']
                   for i in range(self.n_orientations)]

            x = np.arange(self.n_orientations)
            bars = ax2.bar(x - 0.2, expected_oris, 0.4, label='Expected', alpha=0.7)
            bars2 = ax2.bar(x + 0.2, actual_oris, 0.4, label='Learned', alpha=0.7)

            # Color by OSI
            for bar, osi in zip(bars2, osis):
                bar.set_color(plt.cm.viridis(max(0, min(1, osi))))

            ax2.set_xlabel('Neuron Index')
            ax2.set_ylabel('Preferred Orientation (°)')
            ax2.set_title(f'Patch {patch_idx}: Preferred vs Expected')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(f'{save_dir}/tuning_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

    def _plot_orientation_map(self, epoch: int, tuning_data: Dict, save_dir: str):
        """Plot orientation preference map across V1."""
        pref_oris = np.array([d['preferred_ori'] for d in tuning_data['tuning_data']])
        osis = np.array([d['osi'] for d in tuning_data['tuning_data']])

        # Reshape to spatial arrangement
        pref_map = np.zeros((self.n_patches_y, self.n_patches_x, self.n_orientations))
        osi_map = np.zeros((self.n_patches_y, self.n_patches_x, self.n_orientations))

        for patch_idx in range(self.n_patches):
            py = patch_idx // self.n_patches_x
            px = patch_idx % self.n_patches_x
            v1_start = patch_idx * self.n_orientations
            for i in range(self.n_orientations):
                pref_map[py, px, i] = pref_oris[v1_start + i]
                osi_map[py, px, i] = osis[v1_start + i]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Orientation Map - Epoch {epoch}')

        # Mean preferred orientation per patch
        mean_pref = np.mean(pref_map, axis=2)
        im1 = axes[0].imshow(mean_pref, cmap='hsv', vmin=0, vmax=180)
        axes[0].set_title('Mean Preferred Orientation')
        plt.colorbar(im1, ax=axes[0], label='Orientation (°)')

        # Mean OSI per patch
        mean_osi = np.mean(osi_map, axis=2)
        im2 = axes[1].imshow(mean_osi, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Mean OSI')
        plt.colorbar(im2, ax=axes[1])

        # Color-coded orientation map (hue = orientation, saturation = OSI)
        orientation_colors = np.zeros((self.n_patches_y * 2, self.n_patches_x * 2, 3))

        for py in range(self.n_patches_y):
            for px in range(self.n_patches_x):
                # Arrange orientations in a 2x4 or similar grid within each patch
                for i in range(self.n_orientations):
                    iy = (i // 4) if self.n_orientations > 4 else 0
                    ix = i % 4 if self.n_orientations > 4 else i

                    hue = pref_map[py, px, i] / 180.0
                    sat = min(1.0, max(0.3, osi_map[py, px, i]))
                    val = 0.9

                    cy = py * 2 + iy
                    cx = px * 2 + ix
                    if cy < orientation_colors.shape[0] and cx < orientation_colors.shape[1]:
                        orientation_colors[cy, cx] = hsv_to_rgb([hue, sat, val])

        axes[2].imshow(orientation_colors)
        axes[2].set_title('Orientation Map (hue=ori, sat=OSI)')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/orientation_map_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

    def _plot_osi_histogram(self, epoch: int, tuning_data: Dict, save_dir: str):
        """Plot histogram of OSI values."""
        osis = [d['osi'] for d in tuning_data['tuning_data']]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(osis, bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(osis), color='red', linestyle='--',
                   label=f'Mean: {np.mean(osis):.3f}')
        ax.axvline(np.median(osis), color='green', linestyle='--',
                   label=f'Median: {np.median(osis):.3f}')

        ax.set_xlabel('Orientation Selectivity Index (OSI)')
        ax.set_ylabel('Count')
        ax.set_title(f'OSI Distribution - Epoch {epoch}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'{save_dir}/osi_histogram_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

    def _plot_training_history(self, history: Dict, save_dir: str):
        """Plot training metrics over time."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history['epoch'], history['mean_osi'], 'b-o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Mean OSI')
        axes[0].set_title('Orientation Selectivity Over Training')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['epoch'], history['mean_cv'], 'r-o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Circular Variance')
        axes[1].set_title('Tuning Sharpness Over Training')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png', dpi=150)
        plt.close()


def main():
    """Main entry point."""
    print("=" * 70)
    print("Orientation Selectivity Network - Version 2")
    print("RGC (ON/OFF) -> LGN -> V1 Layer 4")
    print("=" * 70)

    network = OrientationSelectivityNetwork(
        retina_size=(16, 16),
        lgn_patch_size=4,
        n_orientations=8,
        max_delay=15,
        initial_bias_strength=0.3,
        stdp_params={
            'tau_pre': 20.0,
            'tau_post': 20.0,
            'A_plus': 0.02,
            'A_minus': 0.025,
            'w_max': 2.5,
            'w_min': 0.01
        }
    )

    orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

    history = network.train(
        n_epochs=25,
        orientations=orientations,
        trials_per_orientation=12,
        trial_duration_ms=350.0,
        visualize_every=5,
        save_dir="training_progress"
    )

    return network, history


if __name__ == "__main__":
    network, history = main()
