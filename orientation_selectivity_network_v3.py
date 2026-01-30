"""
Spiking Neural Network for Orientation Selectivity Emergence - Version 3

Key improvements:
- Stronger structured initial connectivity (Gabor-like receptive fields)
- BCM-like STDP with sliding threshold for homeostatic plasticity
- Enhanced winner-take-all competition with shunting inhibition
- More biologically realistic spike integration

Author: Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from typing import Tuple, Dict, List, Optional
import time
import os

# GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy available - using GPU")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU")


def to_numpy(arr):
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.array(arr)


# Izhikevich parameters
IZHIKEVICH_RS = {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}
IZHIKEVICH_FS = {'a': 0.1, 'b': 0.2, 'c': -65.0, 'd': 2.0}
IZHIKEVICH_TC = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05}


class IzhikevichNeurons:
    """Population of Izhikevich neurons."""

    def __init__(self, n: int, params: dict):
        self.n = n
        self.a, self.b, self.c, self.d = params['a'], params['b'], params['c'], params['d']
        self.v = cp.ones(n) * -65.0
        self.u = cp.ones(n) * self.b * (-65.0)
        self.fired = cp.zeros(n, dtype=bool)

    def step(self, I: cp.ndarray, dt: float = 1.0) -> cp.ndarray:
        for _ in range(2):
            dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * (dt/2)
            self.v += dv
            du = self.a * (self.b * self.v - self.u) * (dt/2)
            self.u += du

        self.fired = self.v >= 30.0
        self.v = cp.where(self.fired, self.c, self.v)
        self.u = cp.where(self.fired, self.u + self.d, self.u)
        return self.fired

    def reset(self):
        self.v = cp.ones(self.n) * -65.0
        self.u = cp.ones(self.n) * self.b * (-65.0)
        self.fired = cp.zeros(self.n, dtype=bool)


class BCMPlasticSynapse:
    """
    BCM-inspired STDP synapse with sliding threshold.

    The key insight: use a homeostatic threshold that depends on
    postsynaptic activity history. This creates competition naturally.
    """

    def __init__(self, n_pre: int, n_post: int,
                 weights: cp.ndarray,
                 connectivity: cp.ndarray,
                 delays: cp.ndarray,
                 w_max: float = 1.0,
                 w_min: float = 0.0,
                 tau_trace: float = 20.0,
                 tau_threshold: float = 1000.0,
                 eta: float = 0.01,
                 max_delay: int = 15):

        self.n_pre = n_pre
        self.n_post = n_post
        self.weights = weights.copy()
        self.connectivity = connectivity.astype(bool)
        self.delays = delays.astype(cp.int32)

        self.w_max = w_max
        self.w_min = w_min
        self.tau_trace = tau_trace
        self.tau_threshold = tau_threshold
        self.eta = eta
        self.max_delay = max_delay

        # Eligibility traces
        self.trace_pre = cp.zeros(n_pre)
        self.trace_post = cp.zeros(n_post)

        # Sliding threshold for BCM (tracks postsynaptic activity)
        self.theta = cp.ones(n_post) * 0.1  # Initial threshold

        # Delay buffer
        self.spike_buffer = cp.zeros((max_delay + 1, n_pre), dtype=bool)
        self.buffer_idx = 0

    def propagate(self, pre_fired: cp.ndarray) -> cp.ndarray:
        self.spike_buffer[self.buffer_idx] = pre_fired

        post_current = cp.zeros(self.n_post)

        for delay in range(self.max_delay + 1):
            buffer_idx = (self.buffer_idx - delay) % (self.max_delay + 1)
            delayed_spikes = self.spike_buffer[buffer_idx]
            delay_mask = (self.delays == delay) & self.connectivity
            contribution = cp.sum(
                self.weights * delay_mask * delayed_spikes[:, None],
                axis=0
            )
            post_current += contribution

        self.buffer_idx = (self.buffer_idx + 1) % (self.max_delay + 1)
        return post_current

    def update(self, pre_fired: cp.ndarray, post_fired: cp.ndarray, dt: float = 1.0):
        """
        BCM-inspired update:
        - If post rate > theta: LTP for active pre synapses
        - If post rate < theta: LTD for active pre synapses
        - Theta slowly tracks post activity
        """
        # Update traces
        decay = cp.exp(-dt / self.tau_trace)
        self.trace_pre = self.trace_pre * decay + pre_fired.astype(cp.float64)
        self.trace_post = self.trace_post * decay + post_fired.astype(cp.float64)

        # Update sliding threshold
        theta_decay = cp.exp(-dt / self.tau_threshold)
        self.theta = self.theta * theta_decay + (1 - theta_decay) * self.trace_post

        # BCM-like plasticity
        # phi = post_activity * (post_activity - theta) -- this creates competition
        post_activity = self.trace_post
        phi = post_activity * (post_activity - self.theta)  # Can be positive or negative

        # Weight change: dw = eta * pre_trace * phi
        dw = self.eta * cp.outer(self.trace_pre, phi)

        # Apply only to connected synapses
        self.weights = self.weights + dw * self.connectivity

        # Clamp
        self.weights = cp.clip(self.weights, self.w_min, self.w_max)

    def reset(self):
        self.trace_pre = cp.zeros(self.n_pre)
        self.trace_post = cp.zeros(self.n_post)
        self.spike_buffer = cp.zeros((self.max_delay + 1, self.n_pre), dtype=bool)
        self.buffer_idx = 0


def create_gabor_rf(size: int, orientation: float, frequency: float = 0.3,
                    sigma: float = None, phase: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create ON and OFF Gabor-like receptive field patterns.

    Returns separate ON and OFF patterns that when combined give oriented RF.
    """
    if sigma is None:
        sigma = size / 4

    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    X, Y = np.meshgrid(x, y)

    theta = np.radians(orientation)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    # Gaussian envelope
    envelope = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))

    # Sinusoidal pattern
    sinusoid = np.cos(2 * np.pi * frequency * Y_rot + phase)

    # Gabor
    gabor = envelope * sinusoid

    # Separate into ON (positive) and OFF (negative) components
    on_pattern = np.maximum(0, gabor)
    off_pattern = np.maximum(0, -gabor)

    # Normalize
    on_pattern = on_pattern / (np.max(on_pattern) + 1e-6)
    off_pattern = off_pattern / (np.max(off_pattern) + 1e-6)

    return on_pattern, off_pattern


class OrientationSelectivityNetworkV3:
    """
    Simplified but effective network for orientation selectivity.

    Key design:
    - Each V1 neuron starts with a Gabor-like RF (strong initial bias)
    - BCM plasticity refines selectivity while maintaining diversity
    - Strong lateral inhibition creates winner-take-all dynamics
    """

    def __init__(self,
                 retina_size: Tuple[int, int] = (16, 16),
                 patch_size: int = 4,
                 n_orientations: int = 8,
                 rf_strength: float = 0.5):

        self.width, self.height = retina_size
        self.patch_size = patch_size
        self.n_orientations = n_orientations
        self.rf_strength = rf_strength

        self.n_patches_x = self.width // patch_size
        self.n_patches_y = self.height // patch_size
        self.n_patches = self.n_patches_x * self.n_patches_y

        self.n_lgn = self.width * self.height  # Per channel
        self.n_v1 = self.n_patches * n_orientations

        print(f"Network: {self.width}x{self.height} LGN -> {self.n_v1} V1 neurons")
        print(f"  {self.n_patches} patches, {n_orientations} orientations each")

        self._init_neurons()
        self._init_synapses()

    def _init_neurons(self):
        self.lgn_on = IzhikevichNeurons(self.n_lgn, IZHIKEVICH_TC)
        self.lgn_off = IzhikevichNeurons(self.n_lgn, IZHIKEVICH_TC)
        self.v1_exc = IzhikevichNeurons(self.n_v1, IZHIKEVICH_RS)
        self.v1_inh = IzhikevichNeurons(self.n_v1 // 2, IZHIKEVICH_FS)

    def _init_synapses(self):
        """Initialize LGN->V1 with Gabor RFs."""

        n_lgn_total = 2 * self.n_lgn  # ON + OFF
        weights = np.zeros((n_lgn_total, self.n_v1))
        connectivity = np.zeros((n_lgn_total, self.n_v1), dtype=bool)
        delays = np.zeros((n_lgn_total, self.n_v1), dtype=int)

        # Orientation angles
        orientations = np.linspace(0, 180, self.n_orientations, endpoint=False)

        for py in range(self.n_patches_y):
            for px in range(self.n_patches_x):
                patch_idx = py * self.n_patches_x + px

                # LGN indices for this patch
                lgn_on_idx = []
                lgn_off_idx = []
                local_coords = []

                for dy in range(self.patch_size):
                    for dx in range(self.patch_size):
                        x = px * self.patch_size + dx
                        y = py * self.patch_size + dy

                        if x < self.width and y < self.height:
                            idx = y * self.width + x
                            lgn_on_idx.append(idx)
                            lgn_off_idx.append(idx + self.n_lgn)
                            local_coords.append((dy, dx))

                # Create RFs for each orientation
                for ori_idx, ori in enumerate(orientations):
                    v1_idx = patch_idx * self.n_orientations + ori_idx

                    # Gabor RF for this orientation
                    on_rf, off_rf = create_gabor_rf(
                        self.patch_size, ori,
                        frequency=0.35,
                        sigma=self.patch_size / 3
                    )

                    # Set weights
                    for i, (lgn_on, lgn_off) in enumerate(zip(lgn_on_idx, lgn_off_idx)):
                        dy, dx = local_coords[i]

                        connectivity[lgn_on, v1_idx] = True
                        connectivity[lgn_off, v1_idx] = True

                        # Strong Gabor-based initial weights
                        base = 0.2 + 0.1 * np.random.random()
                        weights[lgn_on, v1_idx] = base + self.rf_strength * on_rf[dy, dx]
                        weights[lgn_off, v1_idx] = base + self.rf_strength * off_rf[dy, dx]

                        # Variable delays
                        delays[lgn_on, v1_idx] = 2 + np.random.randint(0, 3)
                        delays[lgn_off, v1_idx] = 3 + np.random.randint(0, 3)

        weights = cp.asarray(weights)
        connectivity = cp.asarray(connectivity)
        delays = cp.asarray(delays)

        self.syn_lgn_v1 = BCMPlasticSynapse(
            n_lgn_total, self.n_v1,
            weights, connectivity, delays,
            w_max=1.5, w_min=0.05,
            tau_trace=25.0,
            tau_threshold=500.0,
            eta=0.008,
            max_delay=10
        )

        # Lateral inhibition (V1 exc -> inh -> exc)
        self._init_lateral_inhibition()

    def _init_lateral_inhibition(self):
        """Strong within-patch lateral inhibition."""

        inh_per_patch = (self.n_v1 // 2) // self.n_patches

        # Exc -> Inh
        w_exc_inh = np.zeros((self.n_v1, self.n_v1 // 2))
        conn_exc_inh = np.zeros((self.n_v1, self.n_v1 // 2), dtype=bool)

        # Inh -> Exc
        w_inh_exc = np.zeros((self.n_v1 // 2, self.n_v1))
        conn_inh_exc = np.zeros((self.n_v1 // 2, self.n_v1), dtype=bool)

        for patch_idx in range(self.n_patches):
            exc_start = patch_idx * self.n_orientations
            exc_end = exc_start + self.n_orientations
            inh_start = patch_idx * inh_per_patch
            inh_end = inh_start + inh_per_patch

            # All exc in patch drive all inh in patch
            for ei in range(exc_start, exc_end):
                for ii in range(inh_start, inh_end):
                    conn_exc_inh[ei, ii] = True
                    w_exc_inh[ei, ii] = 0.8 + 0.4 * np.random.random()

            # All inh in patch inhibit all exc in patch
            for ii in range(inh_start, inh_end):
                for ei in range(exc_start, exc_end):
                    conn_inh_exc[ii, ei] = True
                    w_inh_exc[ii, ei] = 3.0 + 1.0 * np.random.random()  # Strong!

        delays_1 = np.ones((self.n_v1, self.n_v1 // 2), dtype=int)
        delays_2 = np.ones((self.n_v1 // 2, self.n_v1), dtype=int) * 2

        self.syn_exc_inh = BCMPlasticSynapse(
            self.n_v1, self.n_v1 // 2,
            cp.asarray(w_exc_inh), cp.asarray(conn_exc_inh), cp.asarray(delays_1),
            w_max=2.0, w_min=0.3, eta=0.001, max_delay=3
        )

        self.syn_inh_exc = BCMPlasticSynapse(
            self.n_v1 // 2, self.n_v1,
            cp.asarray(w_inh_exc), cp.asarray(conn_inh_exc), cp.asarray(delays_2),
            w_max=6.0, w_min=1.5, eta=0.001, max_delay=3
        )

    def generate_grating_spikes(self, orientation: float, time_ms: float,
                                 base_rate: float = 15.0,
                                 max_rate: float = 200.0) -> Tuple[cp.ndarray, cp.ndarray]:
        """Generate LGN spikes for drifting grating stimulus."""

        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        theta = np.radians(orientation)
        sf = 0.2  # spatial frequency
        tf = 3.0  # temporal frequency

        phase = 2 * np.pi * tf * (time_ms / 1000.0)
        spatial = 2 * np.pi * sf * (X * np.cos(theta) + Y * np.sin(theta))

        grating = np.sin(spatial + phase)

        on_rate = base_rate + (max_rate - base_rate) * np.maximum(0, grating)
        off_rate = base_rate + (max_rate - base_rate) * np.maximum(0, -grating)

        on_rate = on_rate.flatten()
        off_rate = off_rate.flatten()

        # Poisson spikes
        on_spikes = np.random.random(self.n_lgn) < (on_rate / 1000.0)
        off_spikes = np.random.random(self.n_lgn) < (off_rate / 1000.0)

        return cp.asarray(on_spikes), cp.asarray(off_spikes)

    def step(self, orientation: float, time_ms: float, learn: bool = True) -> Dict:
        """Single simulation step."""

        # Generate input
        on_spikes, off_spikes = self.generate_grating_spikes(orientation, time_ms)

        # LGN dynamics (driven by input)
        lgn_on_current = on_spikes.astype(cp.float64) * 25.0 + 5.0
        lgn_off_current = off_spikes.astype(cp.float64) * 25.0 + 5.0

        lgn_on_fired = self.lgn_on.step(lgn_on_current)
        lgn_off_fired = self.lgn_off.step(lgn_off_current)

        lgn_combined = cp.concatenate([lgn_on_fired, lgn_off_fired])

        # LGN -> V1
        v1_ff_current = self.syn_lgn_v1.propagate(lgn_combined.astype(cp.float64))

        # Lateral inhibition
        v1_inh_drive = self.syn_exc_inh.propagate(self.v1_exc.fired.astype(cp.float64))
        v1_inh_fired = self.v1_inh.step(v1_inh_drive + 3.0)

        v1_lateral_inh = self.syn_inh_exc.propagate(v1_inh_fired.astype(cp.float64))

        # V1 excitatory
        v1_total_current = v1_ff_current + 4.0 - v1_lateral_inh
        v1_exc_fired = self.v1_exc.step(v1_total_current)

        # Plasticity
        if learn:
            self.syn_lgn_v1.update(lgn_combined, v1_exc_fired)

        return {
            'lgn_on': lgn_on_fired,
            'lgn_off': lgn_off_fired,
            'v1_exc': v1_exc_fired,
            'v1_inh': v1_inh_fired
        }

    def run_trial(self, orientation: float, duration: float = 300.0,
                  learn: bool = True) -> np.ndarray:
        """Run trial, return V1 spike counts."""
        counts = cp.zeros(self.n_v1)
        for t in range(int(duration)):
            spikes = self.step(orientation, float(t), learn=learn)
            counts += spikes['v1_exc'].astype(cp.float64)
        return to_numpy(counts)

    def reset_states(self):
        """Reset neuron and synapse states."""
        self.lgn_on.reset()
        self.lgn_off.reset()
        self.v1_exc.reset()
        self.v1_inh.reset()
        self.syn_lgn_v1.reset()
        self.syn_exc_inh.reset()
        self.syn_inh_exc.reset()

    def measure_tuning(self, orientations: List[float],
                       n_trials: int = 3,
                       duration: float = 250.0) -> Dict:
        """Measure orientation tuning."""

        responses = {ori: [] for ori in orientations}

        for ori in orientations:
            for _ in range(n_trials):
                self.reset_states()
                counts = self.run_trial(ori, duration, learn=False)
                responses[ori].append(counts / (duration / 1000.0))  # Convert to Hz

        # Average
        avg = {ori: np.mean(responses[ori], axis=0) for ori in orientations}

        # Compute metrics
        osis = []
        pref_oris = []
        sorted_oris = sorted(orientations)

        for v1_idx in range(self.n_v1):
            r = np.array([avg[ori][v1_idx] for ori in sorted_oris])

            if np.max(r) > 0:
                pref_idx = np.argmax(r)
                pref_ori = sorted_oris[pref_idx]
                orth_idx = (pref_idx + len(sorted_oris) // 2) % len(sorted_oris)
                r_pref = r[pref_idx]
                r_orth = r[orth_idx]
                osi = (r_pref - r_orth) / (r_pref + r_orth + 1e-6)
            else:
                pref_ori = 0
                osi = 0

            pref_oris.append(pref_ori)
            osis.append(max(0, osi))

        return {
            'avg_responses': avg,
            'osis': np.array(osis),
            'pref_oris': np.array(pref_oris),
            'mean_osi': np.mean(osis)
        }

    def train(self, n_epochs: int = 20,
              orientations: List[float] = None,
              trials_per_ori: int = 10,
              trial_duration: float = 300.0,
              vis_every: int = 5,
              save_dir: str = "training_v3"):
        """Train network."""

        if orientations is None:
            orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

        os.makedirs(save_dir, exist_ok=True)

        print(f"\nTraining: {n_epochs} epochs, {len(orientations)} orientations")
        print(f"  {trials_per_ori} trials/ori, {trial_duration}ms each")

        history = {'epoch': [], 'osi': []}

        # Initial measurement
        print("\nMeasuring initial tuning...")
        tuning = self.measure_tuning(orientations)
        print(f"  Initial OSI: {tuning['mean_osi']:.4f}")
        history['epoch'].append(0)
        history['osi'].append(tuning['mean_osi'])
        self._visualize(0, orientations, tuning, save_dir)

        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            t0 = time.time()

            # Shuffle orientations
            for ori in np.random.permutation(orientations):
                for _ in range(trials_per_ori):
                    self.reset_states()
                    self.run_trial(ori, trial_duration, learn=True)

            print(f"  Time: {time.time() - t0:.1f}s")

            # Measure
            tuning = self.measure_tuning(orientations)
            print(f"  OSI: {tuning['mean_osi']:.4f}")
            history['epoch'].append(epoch)
            history['osi'].append(tuning['mean_osi'])

            if epoch % vis_every == 0 or epoch == n_epochs:
                self._visualize(epoch, orientations, tuning, save_dir)

        # Plot history
        plt.figure(figsize=(8, 4))
        plt.plot(history['epoch'], history['osi'], 'b-o')
        plt.xlabel('Epoch')
        plt.ylabel('Mean OSI')
        plt.title('Orientation Selectivity Over Training')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/training_history.png', dpi=150)
        plt.close()

        print(f"\nDone! Final OSI: {history['osi'][-1]:.4f}")
        return history

    def _visualize(self, epoch: int, orientations: List[float],
                   tuning: Dict, save_dir: str):
        """Generate visualizations."""

        # 1. Receptive fields
        self._plot_rfs(epoch, save_dir)

        # 2. Tuning curves
        self._plot_tuning(epoch, orientations, tuning, save_dir)

        # 3. OSI distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(tuning['osis'], bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
        ax.axvline(tuning['mean_osi'], color='red', linestyle='--',
                   label=f"Mean: {tuning['mean_osi']:.3f}")
        ax.set_xlabel('OSI')
        ax.set_ylabel('Count')
        ax.set_title(f'OSI Distribution - Epoch {epoch}')
        ax.legend()
        plt.savefig(f'{save_dir}/osi_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

    def _plot_rfs(self, epoch: int, save_dir: str):
        """Plot learned receptive fields."""
        weights = to_numpy(self.syn_lgn_v1.weights)

        n_show = min(4, self.n_patches)

        fig, axes = plt.subplots(n_show, self.n_orientations,
                                  figsize=(1.8 * self.n_orientations, 1.8 * n_show))
        fig.suptitle(f'Receptive Fields (ON-OFF) - Epoch {epoch}')

        for pi in range(n_show):
            px = pi % self.n_patches_x
            py = pi // self.n_patches_x

            x0 = px * self.patch_size
            y0 = py * self.patch_size

            for oi in range(self.n_orientations):
                v1_idx = pi * self.n_orientations + oi

                w_on = weights[:self.n_lgn, v1_idx].reshape(self.height, self.width)
                w_off = weights[self.n_lgn:, v1_idx].reshape(self.height, self.width)

                rf = w_on[y0:y0+self.patch_size, x0:x0+self.patch_size] - \
                     w_off[y0:y0+self.patch_size, x0:x0+self.patch_size]

                ax = axes[pi, oi] if n_show > 1 else axes[oi]
                ax.imshow(rf, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
                expected = oi * 180 // self.n_orientations
                ax.set_title(f'{expected}°', fontsize=8)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/rf_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

    def _plot_tuning(self, epoch: int, orientations: List[float],
                     tuning: Dict, save_dir: str):
        """Plot tuning curves."""
        sorted_oris = sorted(orientations)
        avg = tuning['avg_responses']

        n_show = min(4, self.n_patches)

        fig, axes = plt.subplots(n_show, 2, figsize=(12, 3 * n_show))
        fig.suptitle(f'Orientation Tuning - Epoch {epoch}')

        colors = plt.cm.hsv(np.linspace(0, 1, self.n_orientations))

        for pi in range(n_show):
            v1_start = pi * self.n_orientations

            # Tuning curves
            ax1 = axes[pi, 0] if n_show > 1 else axes[0]
            for i in range(self.n_orientations):
                v1_idx = v1_start + i
                r = [avg[ori][v1_idx] for ori in sorted_oris]
                exp = i * 180 // self.n_orientations
                ax1.plot(sorted_oris, r, 'o-', color=colors[i],
                        label=f'{exp}°', alpha=0.7)

            ax1.set_xlabel('Stimulus Orientation (°)')
            ax1.set_ylabel('Firing Rate (Hz)')
            ax1.set_title(f'Patch {pi}')
            ax1.legend(fontsize=6, ncol=2)

            # Preferred vs expected
            ax2 = axes[pi, 1] if n_show > 1 else axes[1]

            expected = [i * 180 // self.n_orientations for i in range(self.n_orientations)]
            actual = [tuning['pref_oris'][v1_start + i] for i in range(self.n_orientations)]
            osis = [tuning['osis'][v1_start + i] for i in range(self.n_orientations)]

            x = np.arange(self.n_orientations)
            ax2.bar(x - 0.2, expected, 0.4, label='Expected', alpha=0.6)
            bars = ax2.bar(x + 0.2, actual, 0.4, label='Learned', alpha=0.8)

            for bar, osi in zip(bars, osis):
                bar.set_color(plt.cm.viridis(min(1, osi)))

            ax2.set_xlabel('Neuron')
            ax2.set_ylabel('Preferred Orientation (°)')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(f'{save_dir}/tuning_epoch_{epoch:03d}.png', dpi=150)
        plt.close()


def main():
    print("=" * 60)
    print("Orientation Selectivity Network V3")
    print("=" * 60)

    net = OrientationSelectivityNetworkV3(
        retina_size=(16, 16),
        patch_size=4,
        n_orientations=8,
        rf_strength=0.6  # Strong initial Gabor bias
    )

    history = net.train(
        n_epochs=20,
        orientations=[0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5],
        trials_per_ori=8,
        trial_duration=300.0,
        vis_every=4,
        save_dir="training_v3"
    )

    return net, history


if __name__ == "__main__":
    net, history = main()
