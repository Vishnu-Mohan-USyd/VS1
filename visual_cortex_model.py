"""
Visual Cortex Spiking Network Model
====================================

A biologically-inspired spiking neural network implementing the hypothesis that
orientation selectivity emerges from:

1. Retinotopic LGN patches projecting to V1 orientation columns
2. STDP-based learning that strengthens orientation-specific connections
3. Lateral competition ensuring diverse orientation preferences

Key Features:
- Izhikevich neurons (biologically realistic spike patterns)
- ON/OFF visual pathways (RGC -> LGN -> V1)
- Gabor-like receptive field initialization
- STDP with soft bounds for stable learning
- Strong lateral inhibition for winner-take-all competition

Usage:
    python visual_cortex_model.py

Author: Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from typing import Tuple, Dict, List, Optional
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GPU/CPU Backend Selection
# ============================================================================

try:
    import cupy as xp
    GPU = True
    print("[GPU] CuPy detected - using GPU acceleration")
except ImportError:
    import numpy as xp
    GPU = False
    print("[CPU] CuPy not found - using NumPy")


def to_cpu(arr):
    """Convert array to CPU numpy."""
    if GPU:
        return xp.asnumpy(arr)
    return np.asarray(arr)


# ============================================================================
# Izhikevich Neuron Parameters (from Izhikevich 2003)
# ============================================================================

# Regular Spiking - excitatory pyramidal cells
RS_PARAMS = {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}

# Fast Spiking - inhibitory basket cells
FS_PARAMS = {'a': 0.1, 'b': 0.2, 'c': -65.0, 'd': 2.0}

# Thalamic - LGN relay cells
TC_PARAMS = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05}


# ============================================================================
# Core Classes
# ============================================================================

class NeuronPopulation:
    """
    Izhikevich neuron population.

    Equations:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        if v >= 30: v = c, u = u + d
    """

    def __init__(self, n: int, params: dict, name: str = ""):
        self.n = n
        self.name = name
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']

        self.reset()

    def reset(self):
        self.v = xp.ones(self.n, dtype=xp.float32) * -65.0
        self.u = xp.ones(self.n, dtype=xp.float32) * (self.b * -65.0)
        self.spikes = xp.zeros(self.n, dtype=bool)

    def update(self, I: xp.ndarray, dt: float = 1.0) -> xp.ndarray:
        """Update neurons with input current I. Returns spike array."""
        # Two half-steps for stability
        for _ in range(2):
            dv = (0.04 * self.v**2 + 5.0 * self.v + 140.0 - self.u + I) * (dt / 2)
            self.v = self.v + dv
            du = self.a * (self.b * self.v - self.u) * (dt / 2)
            self.u = self.u + du

        # Spike detection and reset
        self.spikes = self.v >= 30.0
        self.v = xp.where(self.spikes, self.c, self.v)
        self.u = xp.where(self.spikes, self.u + self.d, self.u)

        return self.spikes


class SynapticProjection:
    """
    Synaptic connections with STDP learning and delays.

    Uses trace-based STDP with soft bounds:
    - Potentiation: dw = A+ * (w_max - w) * pre_trace * post_spike
    - Depression: dw = -A- * (w - w_min) * pre_spike * post_trace
    """

    def __init__(self,
                 n_pre: int,
                 n_post: int,
                 weights: xp.ndarray,
                 mask: xp.ndarray,
                 delays: xp.ndarray = None,
                 w_bounds: Tuple[float, float] = (0.0, 1.0),
                 tau_stdp: float = 20.0,
                 lr_plus: float = 0.01,
                 lr_minus: float = 0.012,
                 max_delay: int = 10):

        self.n_pre = n_pre
        self.n_post = n_post
        self.w = weights.astype(xp.float32)
        self.mask = mask.astype(bool)
        self.w_min, self.w_max = w_bounds
        self.tau = tau_stdp
        self.lr_plus = lr_plus
        self.lr_minus = lr_minus
        self.max_delay = max_delay

        if delays is None:
            self.delays = xp.ones((n_pre, n_post), dtype=xp.int32)
        else:
            self.delays = delays.astype(xp.int32)

        self.reset_traces()

    def reset_traces(self):
        self.pre_trace = xp.zeros(self.n_pre, dtype=xp.float32)
        self.post_trace = xp.zeros(self.n_post, dtype=xp.float32)
        self.spike_buffer = xp.zeros((self.max_delay + 1, self.n_pre), dtype=bool)
        self.buf_idx = 0

    def propagate(self, pre_spikes: xp.ndarray) -> xp.ndarray:
        """Compute postsynaptic current from presynaptic spikes."""
        # Store spikes in delay buffer
        self.spike_buffer[self.buf_idx] = pre_spikes

        # Gather delayed spikes and compute currents
        current = xp.zeros(self.n_post, dtype=xp.float32)

        for d in range(self.max_delay + 1):
            read_idx = (self.buf_idx - d) % (self.max_delay + 1)
            delayed = self.spike_buffer[read_idx]
            delay_mask = (self.delays == d) & self.mask
            current += xp.sum(self.w * delay_mask * delayed[:, None], axis=0)

        self.buf_idx = (self.buf_idx + 1) % (self.max_delay + 1)
        return current

    def learn(self, pre_spikes: xp.ndarray, post_spikes: xp.ndarray, dt: float = 1.0):
        """Apply STDP learning rule."""
        # Update traces
        decay = xp.exp(xp.float32(-dt / self.tau))
        self.pre_trace = self.pre_trace * decay + pre_spikes.astype(xp.float32)
        self.post_trace = self.post_trace * decay + post_spikes.astype(xp.float32)

        # Potentiation: pre active before post
        if xp.any(post_spikes):
            dw_plus = self.lr_plus * (self.w_max - self.w) * xp.outer(
                self.pre_trace, post_spikes.astype(xp.float32))
            self.w = self.w + dw_plus * self.mask

        # Depression: post active before pre
        if xp.any(pre_spikes):
            dw_minus = self.lr_minus * (self.w - self.w_min) * xp.outer(
                pre_spikes.astype(xp.float32), self.post_trace)
            self.w = self.w - dw_minus * self.mask

        self.w = xp.clip(self.w, self.w_min, self.w_max)


def make_gabor_kernel(size: int, theta: float, freq: float = 0.3,
                      sigma: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create ON and OFF Gabor-like weight patterns.

    Args:
        size: Kernel size
        theta: Orientation in degrees
        freq: Spatial frequency
        sigma: Gaussian envelope width

    Returns:
        (on_weights, off_weights) as separate positive patterns
    """
    if sigma is None:
        sigma = size / 3.0

    # Coordinate grid
    c = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    x = x - c
    y = y - c

    # Rotate coordinates
    theta_rad = np.radians(theta)
    x_rot = x * np.cos(theta_rad) + y * np.sin(theta_rad)
    y_rot = -x * np.sin(theta_rad) + y * np.cos(theta_rad)

    # Gabor = Gaussian * Sinusoid
    gaussian = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * freq * y_rot)
    gabor = gaussian * sinusoid

    # Split into ON (positive) and OFF (negative) components
    on_kernel = np.maximum(0, gabor)
    off_kernel = np.maximum(0, -gabor)

    # Normalize
    if on_kernel.max() > 0:
        on_kernel = on_kernel / on_kernel.max()
    if off_kernel.max() > 0:
        off_kernel = off_kernel / off_kernel.max()

    return on_kernel, off_kernel


class VisualCortexModel:
    """
    Complete visual pathway: RGC (ON/OFF) -> LGN -> V1 Layer 4

    Architecture follows the hypothesis:
    - Small LGN patches project to multiple V1 neurons
    - Each V1 neuron has a distinct initial orientation bias
    - STDP refines connections based on stimulus-driven activity
    - Lateral inhibition creates winner-take-all competition
    """

    def __init__(self,
                 visual_field: Tuple[int, int] = (16, 16),
                 patch_size: int = 4,
                 n_orientations: int = 8,
                 gabor_strength: float = 0.8,
                 stdp_rate: float = 0.015):
        """
        Initialize the model.

        Args:
            visual_field: Size of visual input (width, height)
            patch_size: Size of each retinotopic patch
            n_orientations: Number of orientation preferences per patch
            gabor_strength: Initial orientation bias strength (0-1)
            stdp_rate: Base STDP learning rate
        """
        self.width, self.height = visual_field
        self.patch_size = patch_size
        self.n_ori = n_orientations
        self.gabor_strength = gabor_strength

        # Calculate grid dimensions
        self.nx = self.width // patch_size
        self.ny = self.height // patch_size
        self.n_patches = self.nx * self.ny

        # Population sizes
        self.n_lgn = self.width * self.height  # Per channel
        self.n_v1_exc = self.n_patches * n_orientations
        self.n_v1_inh = self.n_patches * max(1, n_orientations // 2)

        # STDP parameters (must be set before building synapses)
        self.stdp_rate = stdp_rate

        print(f"Visual Cortex Model")
        print(f"  Input: {self.width}x{self.height}")
        print(f"  LGN: {self.n_lgn} ON + {self.n_lgn} OFF")
        print(f"  V1: {self.n_v1_exc} excitatory, {self.n_v1_inh} inhibitory")
        print(f"  Patches: {self.nx}x{self.ny} = {self.n_patches}")
        print(f"  Orientations per patch: {n_orientations}")

        # Initialize
        self._build_neurons()
        self._build_lgn_to_v1(gabor_strength)
        self._build_lateral_inhibition()

    def _build_neurons(self):
        """Create neuron populations."""
        self.lgn_on = NeuronPopulation(self.n_lgn, TC_PARAMS, "LGN_ON")
        self.lgn_off = NeuronPopulation(self.n_lgn, TC_PARAMS, "LGN_OFF")
        self.v1_exc = NeuronPopulation(self.n_v1_exc, RS_PARAMS, "V1_Exc")
        self.v1_inh = NeuronPopulation(self.n_v1_inh, FS_PARAMS, "V1_Inh")

    def _build_lgn_to_v1(self, strength: float):
        """
        Build LGN -> V1 connections with Gabor-like initial weights.

        Each V1 neuron receives from one patch with orientation-tuned weights.
        """
        n_lgn_total = 2 * self.n_lgn
        W = np.zeros((n_lgn_total, self.n_v1_exc), dtype=np.float32)
        mask = np.zeros((n_lgn_total, self.n_v1_exc), dtype=bool)
        delays = np.zeros((n_lgn_total, self.n_v1_exc), dtype=np.int32)

        orientations = np.linspace(0, 180, self.n_ori, endpoint=False)

        for py in range(self.ny):
            for px in range(self.nx):
                patch_id = py * self.nx + px

                # LGN indices for this patch
                lgn_on_ids = []
                lgn_off_ids = []
                local_coords = []

                for dy in range(self.patch_size):
                    for dx in range(self.patch_size):
                        gx = px * self.patch_size + dx
                        gy = py * self.patch_size + dy
                        if gx < self.width and gy < self.height:
                            idx = gy * self.width + gx
                            lgn_on_ids.append(idx)
                            lgn_off_ids.append(idx + self.n_lgn)
                            local_coords.append((dy, dx))

                # Create connections for each orientation
                for oi, ori in enumerate(orientations):
                    v1_id = patch_id * self.n_ori + oi

                    # Gabor template (higher frequency for better structure in small patch)
                    on_k, off_k = make_gabor_kernel(
                        self.patch_size, ori,
                        freq=0.5,  # Higher freq for more oriented structure
                        sigma=self.patch_size / 2.0
                    )

                    for i, (on_id, off_id) in enumerate(zip(lgn_on_ids, lgn_off_ids)):
                        ly, lx = local_coords[i]

                        mask[on_id, v1_id] = True
                        mask[off_id, v1_id] = True

                        # Orientation-biased weights (strong Gabor structure)
                        # Use multiplicative rather than additive bias
                        on_w = on_k[ly, lx]
                        off_w = off_k[ly, lx]

                        # Strong connections where Gabor is high, weak where low
                        W[on_id, v1_id] = 0.1 + strength * (on_w ** 0.5)
                        W[off_id, v1_id] = 0.1 + strength * (off_w ** 0.5)

                        # Delays
                        delays[on_id, v1_id] = 2 + np.random.randint(0, 4)
                        delays[off_id, v1_id] = 2 + np.random.randint(0, 4)

        self.syn_lgn_v1 = SynapticProjection(
            n_lgn_total, self.n_v1_exc,
            xp.asarray(W), xp.asarray(mask), xp.asarray(delays),
            w_bounds=(0.02, 1.2),
            tau_stdp=20.0,
            lr_plus=self.stdp_rate,
            lr_minus=self.stdp_rate * 1.2,
            max_delay=8
        )

        n_conn = np.sum(mask)
        print(f"  LGN->V1 connections: {n_conn}")

    def _build_lateral_inhibition(self):
        """Build lateral inhibition: V1_Exc -> V1_Inh -> V1_Exc"""
        inh_per_patch = self.n_v1_inh // self.n_patches

        # Exc -> Inh
        W_ei = np.zeros((self.n_v1_exc, self.n_v1_inh), dtype=np.float32)
        mask_ei = np.zeros((self.n_v1_exc, self.n_v1_inh), dtype=bool)

        # Inh -> Exc
        W_ie = np.zeros((self.n_v1_inh, self.n_v1_exc), dtype=np.float32)
        mask_ie = np.zeros((self.n_v1_inh, self.n_v1_exc), dtype=bool)

        for pid in range(self.n_patches):
            exc_start = pid * self.n_ori
            exc_end = exc_start + self.n_ori
            inh_start = pid * inh_per_patch
            inh_end = inh_start + inh_per_patch

            for ei in range(exc_start, exc_end):
                for ii in range(inh_start, inh_end):
                    mask_ei[ei, ii] = True
                    W_ei[ei, ii] = 1.0 + 0.5 * np.random.random()

            for ii in range(inh_start, inh_end):
                for ei in range(exc_start, exc_end):
                    mask_ie[ii, ei] = True
                    W_ie[ii, ei] = 4.0 + 2.0 * np.random.random()  # Strong inhibition

        self.syn_exc_inh = SynapticProjection(
            self.n_v1_exc, self.n_v1_inh,
            xp.asarray(W_ei), xp.asarray(mask_ei),
            w_bounds=(0.5, 3.0),
            lr_plus=0.002, lr_minus=0.002, max_delay=2
        )

        self.syn_inh_exc = SynapticProjection(
            self.n_v1_inh, self.n_v1_exc,
            xp.asarray(W_ie), xp.asarray(mask_ie),
            w_bounds=(2.0, 10.0),
            lr_plus=0.002, lr_minus=0.002, max_delay=2
        )

    def reset(self):
        """Reset all neuron states and STDP traces."""
        self.lgn_on.reset()
        self.lgn_off.reset()
        self.v1_exc.reset()
        self.v1_inh.reset()
        self.syn_lgn_v1.reset_traces()
        self.syn_exc_inh.reset_traces()
        self.syn_inh_exc.reset_traces()

    def generate_grating(self, orientation: float, t_ms: float,
                         sf: float = 0.2, tf: float = 3.0,
                         contrast: float = 1.0) -> Tuple[xp.ndarray, xp.ndarray]:
        """
        Generate ON/OFF spike rates for a drifting grating.

        Args:
            orientation: Grating orientation in degrees
            t_ms: Time in milliseconds
            sf: Spatial frequency (cycles/pixel)
            tf: Temporal frequency (Hz)
            contrast: Grating contrast (0-1)

        Returns:
            (on_rates, off_rates) in Hz
        """
        x = np.arange(self.width, dtype=np.float32)
        y = np.arange(self.height, dtype=np.float32)
        X, Y = np.meshgrid(x, y)

        theta = np.radians(orientation)
        phase = 2 * np.pi * tf * (t_ms / 1000.0)
        spatial = 2 * np.pi * sf * (X * np.cos(theta) + Y * np.sin(theta))

        grating = contrast * np.sin(spatial + phase)

        base_rate = 20.0
        max_rate = 200.0

        on_rate = base_rate + (max_rate - base_rate) * np.maximum(0, grating)
        off_rate = base_rate + (max_rate - base_rate) * np.maximum(0, -grating)

        return xp.asarray(on_rate.flatten()), xp.asarray(off_rate.flatten())

    def step(self, orientation: float, t_ms: float, learn: bool = True) -> Dict:
        """
        Simulate one millisecond.

        Returns dict with spike counts for each population.
        """
        # Generate input
        on_rate, off_rate = self.generate_grating(orientation, t_ms)

        # Poisson spikes (rate -> probability)
        on_prob = on_rate / 1000.0
        off_prob = off_rate / 1000.0

        on_input = xp.random.random(self.n_lgn) < on_prob
        off_input = xp.random.random(self.n_lgn) < off_prob

        # LGN dynamics
        lgn_on_I = on_input.astype(xp.float32) * 30.0 + 6.0
        lgn_off_I = off_input.astype(xp.float32) * 30.0 + 6.0

        lgn_on_spk = self.lgn_on.update(lgn_on_I)
        lgn_off_spk = self.lgn_off.update(lgn_off_I)

        lgn_combined = xp.concatenate([lgn_on_spk, lgn_off_spk])

        # Feedforward current to V1
        v1_ff = self.syn_lgn_v1.propagate(lgn_combined.astype(xp.float32))

        # Lateral inhibition
        inh_drive = self.syn_exc_inh.propagate(self.v1_exc.spikes.astype(xp.float32))
        v1_inh_spk = self.v1_inh.update(inh_drive + 4.0)
        lateral_inh = self.syn_inh_exc.propagate(v1_inh_spk.astype(xp.float32))

        # V1 excitatory
        v1_exc_I = v1_ff + 5.0 - lateral_inh
        v1_exc_spk = self.v1_exc.update(v1_exc_I)

        # STDP
        if learn:
            self.syn_lgn_v1.learn(lgn_combined, v1_exc_spk)

        return {
            'lgn_on': int(xp.sum(lgn_on_spk)),
            'lgn_off': int(xp.sum(lgn_off_spk)),
            'v1_exc': int(xp.sum(v1_exc_spk)),
            'v1_inh': int(xp.sum(v1_inh_spk)),
            'v1_exc_spikes': v1_exc_spk
        }

    def run_trial(self, orientation: float, duration_ms: int = 300,
                  learn: bool = True) -> np.ndarray:
        """Run a trial and return V1 spike counts."""
        self.reset()
        counts = xp.zeros(self.n_v1_exc, dtype=xp.float32)

        for t in range(duration_ms):
            result = self.step(orientation, float(t), learn=learn)
            counts += result['v1_exc_spikes'].astype(xp.float32)

        return to_cpu(counts)

    def measure_tuning(self, orientations: List[float],
                       n_trials: int = 3,
                       duration: int = 300) -> Dict:
        """
        Measure orientation tuning for all V1 neurons.

        Returns:
            Dict with responses, OSI values, and preferred orientations
        """
        responses = {ori: [] for ori in orientations}

        for ori in orientations:
            for _ in range(n_trials):
                counts = self.run_trial(ori, duration, learn=False)
                rates = counts / (duration / 1000.0)  # Convert to Hz
                responses[ori].append(rates)

        # Average responses
        avg = {ori: np.mean(responses[ori], axis=0) for ori in orientations}

        # Compute OSI and preferred orientation
        sorted_oris = sorted(orientations)
        osis = []
        prefs = []

        for ni in range(self.n_v1_exc):
            r = np.array([avg[o][ni] for o in sorted_oris])

            if r.max() > 1.0:  # Active neuron
                pref_idx = np.argmax(r)
                pref_ori = sorted_oris[pref_idx]
                orth_idx = (pref_idx + len(sorted_oris) // 2) % len(sorted_oris)

                r_pref = r[pref_idx]
                r_orth = r[orth_idx]
                osi = (r_pref - r_orth) / (r_pref + r_orth + 1e-6)
                osi = max(0, osi)  # Clamp to positive
            else:
                pref_ori = 0
                osi = 0

            prefs.append(pref_ori)
            osis.append(osi)

        return {
            'responses': avg,
            'osi': np.array(osis),
            'preferred': np.array(prefs),
            'mean_osi': np.mean(osis)
        }

    def train(self,
              n_epochs: int = 30,
              orientations: List[float] = None,
              trials_per_ori: int = 8,
              duration: int = 300,
              visualize_every: int = 5,
              save_dir: str = "output"):
        """
        Train the network on drifting gratings.

        Args:
            n_epochs: Number of training epochs
            orientations: List of orientations to train on
            trials_per_ori: Trials per orientation per epoch
            duration: Trial duration in ms
            visualize_every: Generate visualizations every N epochs
            save_dir: Directory for output files
        """
        if orientations is None:
            orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Orientations: {orientations}")
        print(f"  Trials/ori: {trials_per_ori}")
        print(f"  Duration: {duration} ms")
        print(f"  Output: {save_dir}/")

        history = {'epoch': [], 'osi': []}

        # Initial measurement
        print(f"\nMeasuring initial tuning...")
        tuning = self.measure_tuning(orientations)
        print(f"  Initial Mean OSI: {tuning['mean_osi']:.4f}")

        history['epoch'].append(0)
        history['osi'].append(tuning['mean_osi'])
        self._visualize(0, orientations, tuning, save_dir)

        # Training loop
        for epoch in range(1, n_epochs + 1):
            t_start = time.time()

            # Shuffle orientation order
            epoch_oris = np.random.permutation(orientations)

            for ori in epoch_oris:
                for _ in range(trials_per_ori):
                    self.run_trial(ori, duration, learn=True)

            t_elapsed = time.time() - t_start

            # Measure tuning
            tuning = self.measure_tuning(orientations)

            history['epoch'].append(epoch)
            history['osi'].append(tuning['mean_osi'])

            print(f"Epoch {epoch:3d}/{n_epochs}: OSI={tuning['mean_osi']:.4f} ({t_elapsed:.1f}s)")

            if epoch % visualize_every == 0 or epoch == n_epochs:
                self._visualize(epoch, orientations, tuning, save_dir)

        # Final history plot
        self._plot_history(history, save_dir)

        print(f"\n{'='*60}")
        print(f"Training Complete")
        print(f"{'='*60}")
        print(f"  Final OSI: {history['osi'][-1]:.4f}")
        print(f"  Change: {history['osi'][-1] - history['osi'][0]:+.4f}")
        print(f"  Results: {save_dir}/")

        return history

    def _visualize(self, epoch: int, orientations: List[float],
                   tuning: Dict, save_dir: str):
        """Generate visualization plots."""

        # 1. Receptive fields
        self._plot_rfs(epoch, save_dir)

        # 2. Tuning curves
        self._plot_tuning_curves(epoch, orientations, tuning, save_dir)

        # 3. OSI histogram
        self._plot_osi_hist(epoch, tuning, save_dir)

        # 4. Orientation map
        self._plot_ori_map(epoch, tuning, save_dir)

    def _plot_rfs(self, epoch: int, save_dir: str):
        """Plot receptive fields (ON-OFF weight patterns)."""
        W = to_cpu(self.syn_lgn_v1.w)

        n_show = min(4, self.n_patches)

        fig, axes = plt.subplots(n_show, self.n_ori,
                                  figsize=(1.5 * self.n_ori, 1.5 * n_show))
        fig.suptitle(f'Receptive Fields - Epoch {epoch}', fontsize=10)

        for pi in range(n_show):
            px = pi % self.nx
            py = pi // self.nx

            x0 = px * self.patch_size
            y0 = py * self.patch_size

            for oi in range(self.n_ori):
                v1_id = pi * self.n_ori + oi

                w_on = W[:self.n_lgn, v1_id].reshape(self.height, self.width)
                w_off = W[self.n_lgn:, v1_id].reshape(self.height, self.width)

                rf = w_on[y0:y0+self.patch_size, x0:x0+self.patch_size] - \
                     w_off[y0:y0+self.patch_size, x0:x0+self.patch_size]

                ax = axes[pi, oi] if n_show > 1 else axes[oi]
                ax.imshow(rf, cmap='RdBu_r', vmin=-0.6, vmax=0.6)
                ax.set_title(f'{oi * 180 // self.n_ori}°', fontsize=8)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/rf_epoch_{epoch:03d}.png', dpi=120, bbox_inches='tight')
        plt.close()

    def _plot_tuning_curves(self, epoch: int, orientations: List[float],
                            tuning: Dict, save_dir: str):
        """Plot orientation tuning curves."""
        sorted_oris = sorted(orientations)
        avg = tuning['responses']

        n_show = min(4, self.n_patches)

        fig, axes = plt.subplots(n_show, 2, figsize=(10, 2.5 * n_show))
        fig.suptitle(f'Orientation Tuning - Epoch {epoch}', fontsize=10)

        colors = plt.cm.hsv(np.linspace(0, 1, self.n_ori))

        for pi in range(n_show):
            v1_start = pi * self.n_ori

            # Left: Tuning curves
            ax1 = axes[pi, 0] if n_show > 1 else axes[0]
            for i in range(self.n_ori):
                v1_id = v1_start + i
                r = [avg[o][v1_id] for o in sorted_oris]
                expected = i * 180 // self.n_ori
                ax1.plot(sorted_oris, r, 'o-', color=colors[i],
                        label=f'{expected}°', alpha=0.8, markersize=4)

            ax1.set_xlabel('Stimulus (°)')
            ax1.set_ylabel('Rate (Hz)')
            ax1.set_title(f'Patch {pi}')
            ax1.legend(fontsize=6, ncol=2, loc='upper right')
            ax1.grid(True, alpha=0.3)

            # Right: Preferred vs expected
            ax2 = axes[pi, 1] if n_show > 1 else axes[1]

            expected_pref = [i * 180 // self.n_ori for i in range(self.n_ori)]
            actual_pref = [tuning['preferred'][v1_start + i] for i in range(self.n_ori)]
            osi_vals = [tuning['osi'][v1_start + i] for i in range(self.n_ori)]

            x = np.arange(self.n_ori)
            ax2.bar(x - 0.2, expected_pref, 0.4, alpha=0.6, label='Expected')
            bars = ax2.bar(x + 0.2, actual_pref, 0.4, alpha=0.8, label='Learned')

            for bar, osi in zip(bars, osi_vals):
                bar.set_color(plt.cm.viridis(min(1, max(0, osi * 2))))

            ax2.set_xlabel('Neuron')
            ax2.set_ylabel('Preferred Ori (°)')
            ax2.legend(fontsize=7)
            ax2.set_xticks(x)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/tuning_epoch_{epoch:03d}.png', dpi=120, bbox_inches='tight')
        plt.close()

    def _plot_osi_hist(self, epoch: int, tuning: Dict, save_dir: str):
        """Plot OSI histogram."""
        fig, ax = plt.subplots(figsize=(7, 3.5))

        osis = tuning['osi']
        ax.hist(osis, bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
        ax.axvline(tuning['mean_osi'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {tuning['mean_osi']:.3f}")
        ax.axvline(np.median(osis), color='blue', linestyle=':', linewidth=2,
                   label=f"Median: {np.median(osis):.3f}")

        ax.set_xlabel('Orientation Selectivity Index')
        ax.set_ylabel('Count')
        ax.set_title(f'OSI Distribution - Epoch {epoch}')
        ax.legend()
        ax.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/osi_epoch_{epoch:03d}.png', dpi=120, bbox_inches='tight')
        plt.close()

    def _plot_ori_map(self, epoch: int, tuning: Dict, save_dir: str):
        """Plot orientation preference map."""
        prefs = tuning['preferred']
        osis = tuning['osi']

        # Create map image
        map_img = np.zeros((self.ny * 2, self.nx * 2, 3))

        for pi in range(self.n_patches):
            px = pi % self.nx
            py = pi // self.nx

            for oi in range(self.n_ori):
                v1_id = pi * self.n_ori + oi

                # Position in map
                ix = oi % (self.n_ori // 2) if self.n_ori > 2 else oi
                iy = oi // (self.n_ori // 2) if self.n_ori > 2 else 0

                mx = px * 2 + ix
                my = py * 2 + iy

                if my < map_img.shape[0] and mx < map_img.shape[1]:
                    hue = prefs[v1_id] / 180.0
                    sat = min(1.0, max(0.3, osis[v1_id] * 2))
                    val = 0.9

                    map_img[my, mx] = hsv_to_rgb([hue, sat, val])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Orientation map
        axes[0].imshow(map_img, interpolation='nearest')
        axes[0].set_title(f'Orientation Map - Epoch {epoch}')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        # OSI map
        osi_map = np.zeros((self.ny, self.nx))
        for pi in range(self.n_patches):
            px = pi % self.nx
            py = pi // self.nx
            v1_start = pi * self.n_ori
            osi_map[py, px] = np.mean(osis[v1_start:v1_start + self.n_ori])

        im = axes[1].imshow(osi_map, cmap='viridis', vmin=0, vmax=0.5)
        axes[1].set_title('Mean OSI per Patch')
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        plt.savefig(f'{save_dir}/ori_map_epoch_{epoch:03d}.png', dpi=120, bbox_inches='tight')
        plt.close()

    def _plot_history(self, history: Dict, save_dir: str):
        """Plot training history."""
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(history['epoch'], history['osi'], 'b-o', linewidth=2, markersize=5)
        ax.fill_between(history['epoch'], 0, history['osi'], alpha=0.2)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Mean OSI', fontsize=11)
        ax.set_title('Orientation Selectivity Over Training', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.5, max(history['osi']) * 1.1))

        # Add annotations
        ax.annotate(f"Initial: {history['osi'][0]:.3f}",
                   xy=(0, history['osi'][0]),
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=9)
        ax.annotate(f"Final: {history['osi'][-1]:.3f}",
                   xy=(history['epoch'][-1], history['osi'][-1]),
                   xytext=(-60, 20), textcoords='offset points',
                   fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png', dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the visual cortex model training."""
    print("\n" + "=" * 60)
    print("Visual Cortex Orientation Selectivity Model")
    print("=" * 60 + "\n")

    # Create model with strong initial orientation bias
    model = VisualCortexModel(
        visual_field=(16, 16),
        patch_size=4,
        n_orientations=8,
        gabor_strength=0.9,  # Strong initial Gabor structure
        stdp_rate=0.015
    )

    # Train
    history = model.train(
        n_epochs=25,
        orientations=[0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5],
        trials_per_ori=12,
        duration=400,
        visualize_every=5,
        save_dir="output"
    )

    return model, history


if __name__ == "__main__":
    model, history = main()
