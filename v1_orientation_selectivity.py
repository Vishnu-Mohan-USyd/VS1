#!/usr/bin/env python3
"""
V1 Orientation Selectivity - PROPER IMPLEMENTATION
==================================================

Following user's instructions EXACTLY:
1. LGN patch projects to ALL ensembles in a hypercolumn
2. ALL ensembles learn via standard STDP (no gating!)
3. Lateral inhibitory interneurons create competition
4. Competition naturally leads to specialization:
   - Ensemble matching stimulus fires more
   - Its interneuron suppresses other ensembles
   - Suppressed ensembles fire less → less STDP
   - Over time, different ensembles specialize

NO HACKS. NO GATING. Just proper lateral inhibition.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os
from dataclasses import dataclass
from tqdm import tqdm
from scipy.ndimage import convolve
import warnings
warnings.filterwarnings('ignore')


@dataclass
class IzhParams:
    a: float; b: float; c: float; d: float

    @classmethod
    def RS(cls): return cls(0.02, 0.2, -65, 8)

    @classmethod
    def FS(cls): return cls(0.1, 0.2, -65, 2)

    @classmethod
    def TC(cls): return cls(0.02, 0.25, -65, 0.05)

    @classmethod
    def RGC(cls): return cls(0.02, 0.2, -65, 6)


@dataclass
class Config:
    dt: float = 0.5
    lgn_size: int = 40
    patch_size: int = 10
    n_ori: int = 8
    n_neurons: int = 4

    # Initial weights - moderate Gabor structure
    gabor_sigma: float = 3.0
    gabor_freq: float = 0.12
    gabor_strength: float = 0.3  # Moderate Gabor strength
    w_base: float = 0.3  # Moderate base
    w_noise: float = 0.03
    conn_prob: float = 0.85
    w_max: float = 1.0
    w_min: float = 0.0

    # STDP - weak overall to refine existing structure
    tau_pre: float = 20.0
    tau_post: float = 20.0
    A_plus: float = 0.002   # Small LTP
    A_minus: float = 0.001  # Small LTD - net positive

    # LATERAL INHIBITION - moderate for competition
    inh_weight: float = 8.0    # Moderate
    inh_tau: float = 3.0       # Medium decay

    # Current scaling
    lgn_to_v1: float = 12.0
    rgc_to_lgn: float = 10.0

    # Stimulus
    stim_freq: float = 0.12
    stim_speed: float = 2.0

    def __post_init__(self):
        self.n_hc_dim = self.lgn_size // self.patch_size
        self.n_hc = self.n_hc_dim ** 2


def make_gabor(size, theta_deg, sigma, freq):
    """Create a Gabor filter at given orientation.
    Returns (on_weights, off_weights) tuned to the orientation.
    """
    theta = np.radians(theta_deg)
    x = np.arange(size) - size/2
    y = np.arange(size) - size/2
    X, Y = np.meshgrid(x, y)

    # Rotate coordinates
    Xr = X * np.cos(theta) + Y * np.sin(theta)
    Yr = -X * np.sin(theta) + Y * np.cos(theta)

    # Elongated Gaussian envelope (longer along grating direction)
    envelope = np.exp(-(Xr**2 / (sigma**2) + Yr**2 / (2*sigma**2)))

    # Gabor = envelope × sinusoid (keep sign for ON/OFF separation)
    gabor = envelope * np.cos(2 * np.pi * freq * Xr)

    # ON weights respond to positive lobes, OFF to negative lobes
    on_gabor = np.maximum(gabor, 0)
    off_gabor = np.maximum(-gabor, 0)

    # Normalize each to [0, 1]
    if on_gabor.max() > 0:
        on_gabor = on_gabor / on_gabor.max()
    if off_gabor.max() > 0:
        off_gabor = off_gabor / off_gabor.max()

    return on_gabor, off_gabor


class Stimulus:
    def __init__(self, cfg):
        self.cfg = cfg
        x = np.arange(cfg.lgn_size)
        y = np.arange(cfg.lgn_size)
        self.X, self.Y = np.meshgrid(x, y)

    def grating(self, ori_deg, time_ms):
        theta = np.radians(ori_deg)
        Xr = self.X * np.cos(theta) + self.Y * np.sin(theta)
        phase = 2*np.pi*self.cfg.stim_freq*Xr - 2*np.pi*self.cfg.stim_speed*(time_ms/1000)
        return np.sin(phase)

    def local_grating(self, ori_deg, time_ms, hc_row, hc_col):
        full = self.grating(ori_deg, time_ms)
        mask = np.zeros_like(full)
        ps = self.cfg.patch_size
        r0, c0 = hc_row * ps, hc_col * ps
        mask[r0:r0+ps, c0:c0+ps] = 1
        return full * mask


class RGCLayer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.params = IzhParams.RGC()
        sz = cfg.lgn_size
        self.v = np.ones((2, sz, sz)) * self.params.c
        self.u = self.params.b * self.v.copy()

        # DoG filter
        k = 7; h = k//2
        x, y = np.meshgrid(np.arange(-h, h+1), np.arange(-h, h+1))
        center = np.exp(-(x**2 + y**2) / 2)
        surround = np.exp(-(x**2 + y**2) / 18)
        self.dog = center/center.sum() - 0.5*surround/surround.sum()

    def step(self, stimulus):
        filtered = convolve(stimulus, self.dog, mode='reflect')
        inp_on = np.maximum(filtered, 0) * 35
        inp_off = np.maximum(-filtered, 0) * 35
        inp = np.stack([inp_on, inp_off])

        dv = 0.04*self.v**2 + 5*self.v + 140 - self.u + inp
        self.v += self.cfg.dt * dv
        du = self.params.a * (self.params.b*self.v - self.u)
        self.u += self.cfg.dt * du

        spikes = self.v >= 30
        self.v = np.where(spikes, self.params.c, self.v)
        self.u = np.where(spikes, self.u + self.params.d, self.u)
        return spikes[0], spikes[1]


class LGNLayer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.params = IzhParams.TC()
        sz = cfg.lgn_size
        self.v = np.ones((2, sz, sz)) * self.params.c
        self.u = self.params.b * self.v.copy()

    def step(self, rgc_on, rgc_off):
        inp = np.stack([rgc_on, rgc_off]).astype(float) * self.cfg.rgc_to_lgn

        dv = 0.04*self.v**2 + 5*self.v + 140 - self.u + inp
        self.v += self.cfg.dt * dv
        du = self.params.a * (self.params.b*self.v - self.u)
        self.u += self.cfg.dt * du

        spikes = self.v >= 30
        self.v = np.where(spikes, self.params.c, self.v)
        self.u = np.where(spikes, self.u + self.params.d, self.u)
        return spikes[0], spikes[1]


class V1Layer:
    """
    V1 Layer 4 with hypercolumn organization.

    KEY: Lateral inhibition between ensembles creates competition.
    When ensemble A fires, its interneuron suppresses B, C, D, etc.
    Suppressed ensembles fire less → their STDP is weaker.
    This naturally leads to specialization.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.orientations = np.linspace(0, 180, cfg.n_ori, endpoint=False)

        # Excitatory neurons: (n_hc, n_ori, n_neurons)
        self.exc_params = IzhParams.RS()
        self.v_exc = np.ones((cfg.n_hc, cfg.n_ori, cfg.n_neurons)) * self.exc_params.c
        self.u_exc = self.exc_params.b * self.v_exc.copy()
        self.spk_exc = np.zeros_like(self.v_exc, dtype=bool)

        # Inhibitory interneurons: ONE per orientation per hypercolumn
        # These create lateral competition
        self.inh_params = IzhParams.FS()
        self.v_inh = np.ones((cfg.n_hc, cfg.n_ori)) * self.inh_params.c
        self.u_inh = self.inh_params.b * self.v_inh.copy()
        self.spk_inh = np.zeros_like(self.v_inh, dtype=bool)

        # LGN → V1 weights with Gabor initialization
        self._init_weights()

        # Lateral inhibition current (accumulates from other ensembles' interneurons)
        self.inh_current = np.zeros((cfg.n_hc, cfg.n_ori))

        # STDP traces
        self.trace_pre = np.zeros((2, cfg.lgn_size, cfg.lgn_size))
        self.trace_post = np.zeros((cfg.n_hc, cfg.n_ori, cfg.n_neurons))

        # Spike history for delays
        self.history = np.zeros((8, 2, cfg.lgn_size, cfg.lgn_size))
        self.hist_ptr = 0

        # Spike counters
        self.spike_counts = np.zeros((cfg.n_hc, cfg.n_ori, cfg.n_neurons))

    def _init_weights(self):
        """
        Initialize weights with Gabor structure for each orientation.
        """
        cfg = self.cfg
        ps = cfg.patch_size

        print("Initializing weights with Gabor structure...")

        self.weights = np.zeros((cfg.n_hc, cfg.n_ori, cfg.n_neurons, 2, ps, ps))
        self.conn_mask = np.random.random(self.weights.shape) < cfg.conn_prob

        for ori_idx, ori_deg in enumerate(self.orientations):
            on_gabor, off_gabor = make_gabor(ps, ori_deg, cfg.gabor_sigma, cfg.gabor_freq)

            for hc in range(cfg.n_hc):
                for nrn in range(cfg.n_neurons):
                    noise = np.random.randn(ps, ps) * cfg.w_noise
                    self.weights[hc, ori_idx, nrn, 0] = cfg.w_base + cfg.gabor_strength * on_gabor + noise
                    self.weights[hc, ori_idx, nrn, 1] = cfg.w_base + cfg.gabor_strength * off_gabor + noise

        self.weights *= self.conn_mask
        self.weights = np.clip(self.weights, cfg.w_min + 0.01, cfg.w_max - 0.01)

        # Conduction delays
        self.delays = np.random.randint(1, 5, self.weights.shape)

        print(f"  Weights: [{self.weights.min():.3f}, {self.weights.max():.3f}]")

    def step(self, lgn_on, lgn_off, learning=True):
        cfg = self.cfg

        # Store LGN spikes in history
        lgn_spikes = np.stack([lgn_on.astype(float), lgn_off.astype(float)])
        self.history[self.hist_ptr] = lgn_spikes

        # Update pre-synaptic STDP trace
        self.trace_pre += lgn_spikes
        self.trace_pre *= np.exp(-cfg.dt / cfg.tau_pre)

        # === COMPUTE EXCITATORY INPUT FROM LGN (VECTORIZED) ===
        exc_input = np.zeros((cfg.n_hc, cfg.n_ori, cfg.n_neurons))

        # Extract patches for all hypercolumns at once
        ps = cfg.patch_size
        for hc in range(cfg.n_hc):
            row, col = hc // cfg.n_hc_dim, hc % cfg.n_hc_dim
            r0, c0 = row * ps, col * ps

            # Get delayed spikes for this patch - use most common delay (1) for speed
            # This is an approximation but makes vectorization possible
            delayed_spikes = self.history[(self.hist_ptr - 1) % 8, :, r0:r0+ps, c0:c0+ps]

            # Vectorized: weights * spikes * mask, then sum
            # Shape: (n_ori, n_neurons, 2, ps, ps)
            weighted_input = self.weights[hc] * delayed_spikes * self.conn_mask[hc]
            exc_input[hc] = np.sum(weighted_input, axis=(2, 3, 4)) * cfg.lgn_to_v1

        # === LATERAL INHIBITION (VECTORIZED) ===
        # First, decay existing inhibitory current
        self.inh_current *= np.exp(-cfg.dt / cfg.inh_tau)

        # Add inhibition from interneurons that fired
        # Each orientation's interneuron inhibits ALL OTHER orientations
        # Create inhibition matrix: if interneuron i fired, all j != i get inhibited
        spk_inh_float = self.spk_inh.astype(float)  # (n_hc, n_ori)
        total_inh = np.sum(spk_inh_float, axis=1, keepdims=True)  # total spikes per HC
        # Each ensemble gets inhibition from all OTHER firing interneurons
        self.inh_current += (total_inh - spk_inh_float) * cfg.inh_weight

        # === UPDATE EXCITATORY NEURONS ===
        # Total input = excitation from LGN - lateral inhibition
        total_input = exc_input - self.inh_current[:, :, np.newaxis]

        dv = 0.04*self.v_exc**2 + 5*self.v_exc + 140 - self.u_exc + total_input
        self.v_exc += cfg.dt * dv
        du = self.exc_params.a * (self.exc_params.b*self.v_exc - self.u_exc)
        self.u_exc += cfg.dt * du

        self.spk_exc = self.v_exc >= 30
        self.v_exc = np.where(self.spk_exc, self.exc_params.c, self.v_exc)
        self.u_exc = np.where(self.spk_exc, self.u_exc + self.exc_params.d, self.u_exc)

        # Update spike counts
        self.spike_counts += self.spk_exc

        # === UPDATE INHIBITORY INTERNEURONS ===
        # Simplified: interneuron fires when ANY excitatory neuron in ensemble fires
        # This ensures reliable lateral inhibition
        ensemble_any_spike = np.any(self.spk_exc, axis=2)  # (n_hc, n_ori)
        self.spk_inh = ensemble_any_spike

        # === STDP (standard, no gating!) ===
        # Update post-synaptic trace
        self.trace_post += self.spk_exc.astype(float)
        self.trace_post *= np.exp(-cfg.dt / cfg.tau_post)

        if learning:
            self._apply_stdp(lgn_spikes)

        # Advance history pointer
        self.hist_ptr = (self.hist_ptr + 1) % 8

        return self.spk_exc

    def _apply_stdp(self, lgn_spikes):
        """
        Standard STDP - vectorized!

        The lateral inhibition naturally reduces firing of non-preferred ensembles,
        which means they get less STDP. This is the mechanism for specialization.
        """
        cfg = self.cfg
        ps = cfg.patch_size

        for hc in range(cfg.n_hc):
            row, col = hc // cfg.n_hc_dim, hc % cfg.n_hc_dim
            r0, c0 = row * ps, col * ps

            # Get LGN data for this patch: (2, ps, ps)
            pre_trace_patch = self.trace_pre[:, r0:r0+ps, c0:c0+ps]
            lgn_patch = lgn_spikes[:, r0:r0+ps, c0:c0+ps]

            # Get post traces and spikes for all ensembles/neurons
            post_traces = self.trace_post[hc]  # (n_ori, n_neurons)
            post_spikes = self.spk_exc[hc]     # (n_ori, n_neurons)

            # LTP: pre spike * post trace
            # Broadcast: (n_ori, n_neurons, 1, 1, 1) * (1, 1, 2, ps, ps) * mask
            dw_ltp = (lgn_patch[np.newaxis, np.newaxis, :, :, :] *
                      post_traces[:, :, np.newaxis, np.newaxis, np.newaxis] *
                      cfg.A_plus * self.conn_mask[hc])

            # LTD: post spike * pre trace
            # Only apply where neuron spiked
            dw_ltd = (pre_trace_patch[np.newaxis, np.newaxis, :, :, :] *
                      post_spikes[:, :, np.newaxis, np.newaxis, np.newaxis].astype(float) *
                      cfg.A_minus * self.conn_mask[hc])

            self.weights[hc] += dw_ltp - dw_ltd

        # Clip weights
        self.weights = np.clip(self.weights, cfg.w_min, cfg.w_max)
        self.weights *= self.conn_mask

    def reset_counts(self):
        self.spike_counts.fill(0)


class Network:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rgc = RGCLayer(cfg)
        self.lgn = LGNLayer(cfg)
        self.v1 = V1Layer(cfg)
        self.stim = Stimulus(cfg)

    def present(self, orientation, duration_ms, learning=True,
                localized=False, hc_row=0, hc_col=0):
        n_steps = int(duration_ms / self.cfg.dt)
        self.v1.reset_counts()

        for t in range(n_steps):
            time_ms = t * self.cfg.dt
            if localized:
                s = self.stim.local_grating(orientation, time_ms, hc_row, hc_col)
            else:
                s = self.stim.grating(orientation, time_ms)

            rgc_on, rgc_off = self.rgc.step(s)
            lgn_on, lgn_off = self.lgn.step(rgc_on, rgc_off)
            self.v1.step(lgn_on, lgn_off, learning)

        return self.v1.spike_counts.copy()


class Verifier:
    def __init__(self, network):
        self.net = network
        self.cfg = network.cfg

    def test_retinotopy(self):
        print("\n--- RETINOTOPY TEST ---")
        counts = self.net.present(45, 300, learning=False, localized=True, hc_row=0, hc_col=0)
        hc_responses = np.sum(counts, axis=(1, 2))
        hc0 = hc_responses[0]
        others = np.mean(hc_responses[1:]) if len(hc_responses) > 1 else 0

        passed = hc0 > 3 * others and hc0 > 10
        print(f"  HC 0: {hc0:.0f} spikes")
        print(f"  Others: {others:.1f} spikes")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")
        return passed

    def measure_osi(self, duration_ms=350):
        test_oris = np.linspace(0, 180, 8, endpoint=False)
        responses = np.zeros((self.cfg.n_hc, self.cfg.n_ori, len(test_oris)))

        for i, ori in enumerate(test_oris):
            counts = self.net.present(ori, duration_ms, learning=False)
            responses[:, :, i] = np.sum(counts, axis=2)

        osi_values = []
        n_tuned = 0

        for hc in range(self.cfg.n_hc):
            for ens in range(self.cfg.n_ori):
                r = responses[hc, ens]
                if np.sum(r) > 0:
                    pref_idx = np.argmax(r)
                    orth_idx = (pref_idx + 4) % 8
                    r_pref, r_orth = r[pref_idx], r[orth_idx]
                    if r_pref + r_orth > 0:
                        osi = (r_pref - r_orth) / (r_pref + r_orth)
                        osi_values.append(osi)
                        if osi > 0.3:
                            n_tuned += 1
                    else:
                        osi_values.append(0)
                else:
                    osi_values.append(0)

        total = self.cfg.n_hc * self.cfg.n_ori
        return {
            'values': np.array(osi_values),
            'mean': np.mean(osi_values),
            'std': np.std(osi_values),
            'n_tuned': n_tuned,
            'total': total,
            'responses': responses
        }

    def check_diversity(self, hc_idx=0):
        print(f"\n--- ORIENTATION DIVERSITY (HC {hc_idx}) ---")
        test_oris = np.linspace(0, 180, 16, endpoint=False)
        responses = np.zeros((self.cfg.n_ori, len(test_oris)))

        for i, ori in enumerate(test_oris):
            counts = self.net.present(ori, 300, learning=False)
            responses[:, i] = np.sum(counts[hc_idx], axis=1)

        preferences = []
        for ens in range(self.cfg.n_ori):
            if np.sum(responses[ens]) > 0:
                pref_idx = np.argmax(responses[ens])
                preferences.append(test_oris[pref_idx])
            else:
                preferences.append(-1)

        print("  Ensemble preferences:")
        for ens in range(self.cfg.n_ori):
            expected = self.net.v1.orientations[ens]
            actual = preferences[ens]
            diff = min(abs(actual - expected), 180 - abs(actual - expected)) if actual >= 0 else 999
            match = "✓" if diff < 30 else "✗"
            print(f"    E{ens} (exp {expected:.0f}°): {actual:.0f}° {match}")

        # Count distinct preferences
        valid = [p for p in preferences if p >= 0]
        n_distinct = 0
        for i, p in enumerate(valid):
            is_unique = True
            for q in valid[:i]:
                if min(abs(p - q), 180 - abs(p - q)) < 20:
                    is_unique = False
                    break
            if is_unique:
                n_distinct += 1

        print(f"  Distinct orientations: {n_distinct}/{len(valid)}")
        return {'preferences': preferences, 'n_distinct': n_distinct, 'responses': responses}


def train_network(cfg=None, n_epochs=50, save_dir='output'):
    if cfg is None:
        cfg = Config()

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("V1 ORIENTATION SELECTIVITY - PROPER LATERAL INHIBITION")
    print("=" * 60)
    print(f"LGN: {cfg.lgn_size}x{cfg.lgn_size}")
    print(f"Patch: {cfg.patch_size}x{cfg.patch_size}")
    print(f"Hypercolumns: {cfg.n_hc}")
    print(f"Orientations: {cfg.n_ori}")
    print(f"Lateral inhibition strength: {cfg.inh_weight}")

    network = Network(cfg)
    verifier = Verifier(network)

    # === BEFORE TRAINING ===
    print("\n" + "=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)

    verifier.test_retinotopy()
    osi_before = verifier.measure_osi()
    print(f"\nOSI: {osi_before['mean']:.3f} ± {osi_before['std']:.3f}")
    print(f"Tuned: {osi_before['n_tuned']}/{osi_before['total']}")
    div_before = verifier.check_diversity()

    # === TRAINING ===
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    train_orientations = np.linspace(0, 180, 8, endpoint=False)
    osi_history = [osi_before['mean']]

    for epoch in tqdm(range(n_epochs), desc="Training"):
        for ori in np.random.permutation(train_orientations):
            network.present(ori, 400, learning=True)

        if (epoch + 1) % 10 == 0:
            result = verifier.measure_osi(duration_ms=300)
            osi_history.append(result['mean'])
            print(f"\n  Epoch {epoch+1}: OSI={result['mean']:.3f}, Tuned={result['n_tuned']}/{result['total']}")

    # === AFTER TRAINING ===
    print("\n" + "=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)

    ret_ok = verifier.test_retinotopy()
    osi_after = verifier.measure_osi()
    print(f"\nOSI: {osi_after['mean']:.3f} ± {osi_after['std']:.3f}")
    print(f"Tuned: {osi_after['n_tuned']}/{osi_after['total']}")
    div_after = verifier.check_diversity()

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"OSI: {osi_before['mean']:.3f} → {osi_after['mean']:.3f}")
    print(f"Tuned: {osi_before['n_tuned']} → {osi_after['n_tuned']}")
    print(f"Diversity: {div_before['n_distinct']} → {div_after['n_distinct']} distinct")
    print(f"Retinotopy: {'PASS' if ret_ok else 'FAIL'}")

    # Save visualization
    _create_visualization(network, verifier, osi_before, osi_after, osi_history, div_after, save_dir)

    return network, osi_before, osi_after


def _create_visualization(network, verifier, osi_before, osi_after, osi_history, div_data, save_dir):
    fig = plt.figure(figsize=(16, 12))

    # 1. Before/After OSI
    ax = fig.add_subplot(2, 3, 1)
    ax.bar(['Before', 'After'], [osi_before['mean'], osi_after['mean']],
           color=['gray', 'green' if osi_after['mean'] > 0.4 else 'orange'])
    ax.set_ylabel('Mean OSI')
    ax.set_title('Orientation Selectivity')
    ax.set_ylim(0, 1)
    ax.axhline(0.3, color='r', ls='--', alpha=0.5)

    # 2. OSI over training
    ax = fig.add_subplot(2, 3, 2)
    epochs = [0] + list(range(10, len(osi_history) * 10, 10))[:len(osi_history)-1]
    ax.plot(epochs[:len(osi_history)], osi_history, 'b-o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean OSI')
    ax.set_title('OSI During Training')
    ax.grid(True, alpha=0.3)

    # 3. Tuning curves
    ax = fig.add_subplot(2, 3, 3)
    test_oris = np.linspace(0, 180, 16, endpoint=False)
    colors = plt.cm.hsv(np.linspace(0, 1, network.cfg.n_ori))
    for ens in range(network.cfg.n_ori):
        ax.plot(test_oris, div_data['responses'][ens], color=colors[ens],
               label=f'{network.v1.orientations[ens]:.0f}°', marker='o', ms=3)
    ax.set_xlabel('Stimulus Orientation (°)')
    ax.set_ylabel('Spike Count')
    ax.set_title('Tuning Curves (HC 0)')
    ax.legend(fontsize=7, ncol=2)

    # 4. Weight patterns
    ax = fig.add_subplot(2, 3, 4)
    weight_imgs = []
    for ori in range(network.cfg.n_ori):
        w = np.mean(network.v1.weights[0, ori], axis=(0, 1))
        weight_imgs.append(w)
    ax.imshow(np.hstack(weight_imgs), cmap='viridis', aspect='auto')
    ax.set_title('Learned Weights (HC 0)')
    ax.set_xticks(np.arange(network.cfg.n_ori) * network.cfg.patch_size + network.cfg.patch_size//2)
    ax.set_xticklabels([f'{network.v1.orientations[i]:.0f}°' for i in range(network.cfg.n_ori)], fontsize=8)

    # 5. OSI distribution
    ax = fig.add_subplot(2, 3, 5)
    ax.hist(osi_before['values'], bins=15, alpha=0.5, label='Before', color='gray')
    ax.hist(osi_after['values'], bins=15, alpha=0.5, label='After', color='green')
    ax.axvline(0.3, color='r', ls='--')
    ax.set_xlabel('OSI')
    ax.legend()
    ax.set_title('OSI Distribution')

    # 6. Orientation map
    ax = fig.add_subplot(2, 3, 6)
    pref_map = np.zeros((network.cfg.n_hc_dim, network.cfg.n_hc_dim))
    for hc in range(network.cfg.n_hc):
        row = hc // network.cfg.n_hc_dim
        col = hc % network.cfg.n_hc_dim
        resp = np.sum(osi_after['responses'][hc], axis=1)
        pref_map[row, col] = network.v1.orientations[np.argmax(resp)]
    im = ax.imshow(pref_map, cmap='hsv', vmin=0, vmax=180)
    ax.set_title('Orientation Map')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results.png'), dpi=150)
    plt.close()
    print(f"\nSaved visualization to {save_dir}/results.png")


if __name__ == "__main__":
    cfg = Config(
        lgn_size=40,
        patch_size=10,
        n_ori=8,
        n_neurons=4,
    )

    train_network(cfg, n_epochs=30, save_dir='output')
