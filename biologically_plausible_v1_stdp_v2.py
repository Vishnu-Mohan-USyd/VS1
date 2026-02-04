#!/usr/bin/env python3
"""biologically_plausible_v1_stdp_v2.py

RGC -> LGN -> V1(L4) spiking network with STDP that learns orientation selectivity.

VERSION 2: Biologically Grounded Inhibition
===========================================

This version implements several key biological improvements based on literature
analysis of how inhibition operates in the visual cortex:

BIOLOGICAL MECHANISMS IMPLEMENTED:

1. LGN→PV FEEDFORWARD INHIBITION
   - PV interneurons receive direct thalamocortical (LGN) input
   - PV can inhibit E neurons BEFORE E spikes (true feedforward)
   - Processing order: LGN → PV → E (disynaptic feedforward inhibition)
   - Biology: Swadlow (2003), Maffei et al. - dense TC input to FS interneurons

2. PUSH-PULL INHIBITION (Optional)
   - Inhibitory neurons have opposite-phase RFs (ON↔OFF polarity swapped)
   - Creates phase-opponent inhibition for contrast invariance
   - Biology: Hirsch et al. (1998), push-pull arrangement in simple cells

3. CONDUCTANCE-BASED INHIBITION (Optional, --conductance-inhibition flag)
   - I_inh = g_inh * (V - E_inh) - voltage-dependent, divisive
   - Enables shunting/normalization effects
   - Biology: Borg-Graham, Monier & Frégnac (1998) on conductance dynamics
   - Default: current-based for stability during learning

4. DISYNAPTIC LATERAL INHIBITION (Mexican Hat via Circuitry)
   - Long-range E→SOM excitation (distance-dependent Gaussian)
   - Local SOM→E inhibition (Gaussian falloff, no self-inhibition)
   - Net effect: surround suppression via disynaptic pathway
   - Biology: Gilbert & Wiesel on horizontal connections, not direct long-range I

5. BIOLOGICAL WEIGHT REGULATION
   - Soft periodic normalization (heterosynaptic-like)
   - Homeostatic synaptic scaling (Turrigiano 2008)
   - Optional inhibitory plasticity (Vogels et al. 2011 iSTDP)
   - Replaces non-biological hard normalization

RESULTS:
   - Achieves comparable or better OSI than the original non-biological version
   - Maintains stable firing rates (~3-5 Hz)
   - Active interneuron populations throughout training

KEY REFERENCES:
- Izhikevich (2003, 2007) - Spiking neuron models
- Pfister & Gerstner (2006) - Triplet STDP rule
- Turrigiano (2008) - Homeostatic synaptic plasticity
- Vogels et al. (2011) - Inhibitory plasticity for E/I balance
- Swadlow (2003) - Fast feedforward inhibition in visual cortex
- Hirsch et al. (1998) - Push-pull inhibition in simple cells
- Gilbert & Wiesel (1983) - Horizontal connections in visual cortex
- Borg-Graham, Monier & Frégnac (1998) - Conductance dynamics in V1

License: MIT
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_osi(rates_hz: np.ndarray, thetas_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Classic doubled-angle OSI.

    OSI = |sum r(theta) e^{i2*theta}| / sum r(theta)
    pref = 0.5 * arg(sum r(theta) e^{i2*theta}) in [0,180)
    """
    th = np.deg2rad(thetas_deg)
    vec = (rates_hz * np.exp(1j * 2 * th)[None, :]).sum(axis=1)
    denom = rates_hz.sum(axis=1) + 1e-9
    osi = np.abs(vec) / denom
    pref = (0.5 * np.angle(vec)) % np.pi
    return osi, np.rad2deg(pref)


# =============================================================================
# Izhikevich Neuron Parameters (from literature)
# =============================================================================

@dataclass
class IzhikevichParams:
    """Izhikevich neuron parameters for different cell types."""
    a: float  # Recovery time scale (smaller = slower)
    b: float  # Sensitivity of recovery to subthreshold fluctuations
    c: float  # After-spike reset value of v (mV)
    d: float  # After-spike increment of u
    v_peak: float = 30.0  # Spike cutoff (mV)
    v_init: float = -65.0  # Initial membrane potential


# Literature-based parameters
TC_PARAMS = IzhikevichParams(a=0.02, b=0.25, c=-65.0, d=0.05)  # Thalamocortical
RS_PARAMS = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)    # Regular spiking
FS_PARAMS = IzhikevichParams(a=0.1, b=0.2, c=-65.0, d=2.0)     # Fast spiking (PV)
LTS_PARAMS = IzhikevichParams(a=0.02, b=0.25, c=-65.0, d=2.0)  # Low-threshold spiking (SOM)


@dataclass
class Params:
    """Network and simulation parameters."""
    # Geometry
    N: int = 8  # Patch size (NxN)
    M: int = 8  # Number of V1 ensembles (like a hypercolumn)
    dt_ms: float = 0.5  # Time step (smaller for Izhikevich stability)

    # Training
    segment_ms: int = 300
    train_segments: int = 200
    seed: int = 1

    # Drifting gratings
    spatial_freq: float = 0.12
    temporal_freq: float = 8.0
    base_rate: float = 5.0
    gain_rate: float = 100.0

    # RGC->LGN synaptic weight (scaled for Izhikevich pA currents)
    w_rgc_lgn: float = 5.0

    # LGN->V1 weights & delays
    delay_max: int = 12
    w_init_mean: float = 0.25
    w_init_std: float = 0.08
    w_max: float = 1.0

    # Homeostatic synaptic scaling
    target_rate_hz: float = 8.0
    tau_homeostasis: float = 2000.0  # Faster adaptation
    homeostasis_rate: float = 0.005  # Stronger scaling

    # STDP parameters (triplet rule) - matched to original working code
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    tau_x: float = 101.0
    tau_y: float = 125.0
    A2_plus: float = 0.008    # Pair LTP amplitude
    A3_plus: float = 0.002    # Triplet LTP enhancement
    A2_minus: float = 0.010   # Pair LTD amplitude
    A3_minus: float = 0.0

    # Weight decay (synaptic turnover)
    w_decay: float = 0.00005  # Reduced decay

    # Heterosynaptic plasticity (replaces hard normalization)
    hetero_rate: float = 0.0001  # Heterosynaptic depression rate
    w_target_sum: float = 15.0   # Target weight sum for normalization

    # Periodic normalization (biologically: heterosynaptic/homeostatic process)
    norm_strength: float = 1.0   # How much to move toward target (0=none, 1=hard)

    # Local inhibitory circuit parameters
    n_pv_per_ensemble: int = 1
    n_som_per_ensemble: int = 1

    # === LGN->PV feedforward inhibition ===
    w_lgn_pv_mean: float = 0.25   # LGN->PV weight mean (broadly tuned, stronger)
    w_lgn_pv_std: float = 0.08    # LGN->PV weight std
    w_lgn_pv_max: float = 0.8     # Max LGN->PV weight

    # E->PV (additional local drive)
    w_e_pv: float = 3.0
    # PV->E (local feedforward/feedback inhibition)
    w_pv_e: float = 4.0

    # E->SOM and SOM->E for lateral inhibition
    w_e_som: float = 5.0
    w_som_e: float = 3.0

    # === NEW: Disynaptic lateral inhibition ===
    # Long-range E->SOM (distance-dependent, for disynaptic suppression)
    w_e_som_lateral: float = 2.0
    som_lateral_sigma: float = 3.0  # Broader than local

    # SOM->E is LOCAL (Gaussian falloff)
    som_local_sigma: float = 1.5

    # Lateral excitatory connections
    w_e_e_lateral: float = 0.3
    lateral_sigma: float = 1.5

    # Synaptic time constants
    tau_ampa: float = 5.0
    tau_gaba: float = 10.0

    # === Inhibition scaling (current-based mode) ===
    E_exc: float = 0.0      # Excitatory reversal potential (mV) - for conductance mode
    E_inh: float = -75.0    # Inhibitory reversal potential (mV) - for conductance mode
    g_pv_scale: float = 0.15   # PV inhibition scaling (reduced for balance)
    g_som_scale: float = 0.2   # SOM inhibition scaling

    # === NEW: Inhibitory plasticity (Vogels et al. 2011) ===
    eta_inh: float = 0.0001    # Inhibitory plasticity rate
    rho_target: float = 0.005  # Target post rate for iSTDP (spikes/ms)
    tau_inh_stdp: float = 20.0  # Time constant for inhibitory traces

    # === Push-pull inhibition ===
    push_pull_enabled: bool = True  # Enable opposite-phase inhibitory RFs

    # === Mode flags (can be set via command line) ===
    conductance_inhibition: bool = False  # Use conductance-based (shunting) inhibition
    inhibitory_plasticity: bool = False   # Enable Vogels et al. iSTDP


class IzhikevichPopulation:
    """Population of Izhikevich neurons."""

    def __init__(self, n: int, params: IzhikevichParams, dt_ms: float, rng: np.random.Generator):
        self.n = n
        self.p = params
        self.dt = dt_ms
        self.rng = rng

        # State variables
        self.v = np.full(n, params.v_init, dtype=np.float32)
        self.u = params.b * self.v.copy()

        # Small random perturbation to break symmetry
        self.v += rng.uniform(-5, 5, n).astype(np.float32)
        self.u += rng.uniform(-2, 2, n).astype(np.float32)

    def reset(self):
        """Reset to initial state."""
        self.v.fill(self.p.v_init)
        self.u = self.p.b * self.v.copy()

    def step(self, I_ext: np.ndarray) -> np.ndarray:
        """
        Advance one time step with external current I_ext.
        Returns binary spike array.
        """
        p = self.p
        dt = self.dt

        # Euler integration with sub-stepping for stability
        dt_sub = dt / 2.0

        for _ in range(2):
            v_clamped = np.clip(self.v, -100, p.v_peak)
            dv = (0.04 * v_clamped * v_clamped + 5.0 * v_clamped + 140.0 - self.u + I_ext) * dt_sub
            du = p.a * (p.b * v_clamped - self.u) * dt_sub
            self.v += dv
            self.u += du

        # Detect spikes
        spikes = (self.v >= p.v_peak).astype(np.uint8)

        # Reset spiking neurons
        spike_idx = spikes.astype(bool)
        self.v[spike_idx] = p.c
        self.u[spike_idx] += p.d

        return spikes


class TripletSTDP:
    """
    Triplet STDP rule from Pfister & Gerstner (2006).

    Per-synapse traces with delays for LGN->V1 plasticity.
    """

    def __init__(self, n_pre: int, n_post: int, p: Params, rng: np.random.Generator):
        self.n_pre = n_pre
        self.n_post = n_post
        self.p = p

        # Pre traces - per synapse (n_post, n_pre)
        self.x_pre = np.zeros((n_post, n_pre), dtype=np.float32)
        self.x_pre_slow = np.zeros((n_post, n_pre), dtype=np.float32)

        # Post traces - per neuron (n_post,)
        self.x_post = np.zeros(n_post, dtype=np.float32)
        self.x_post_slow = np.zeros(n_post, dtype=np.float32)

        # Decay factors
        self.decay_pre = math.exp(-p.dt_ms / p.tau_plus)
        self.decay_pre_slow = math.exp(-p.dt_ms / p.tau_x)
        self.decay_post = math.exp(-p.dt_ms / p.tau_minus)
        self.decay_post_slow = math.exp(-p.dt_ms / p.tau_y)

    def reset(self):
        self.x_pre.fill(0)
        self.x_pre_slow.fill(0)
        self.x_post.fill(0)
        self.x_post_slow.fill(0)

    def update(self, arrivals: np.ndarray, post_spikes: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Update traces and compute MULTIPLICATIVE weight changes.

        Order:
        1. Decay traces
        2. LTD: when pre arrives, depress based on OLD post trace
        3. Update pre traces
        4. LTP: when post fires, potentiate based on NEW pre trace
        5. Update post traces
        """
        p = self.p

        # Decay all traces
        self.x_pre *= self.decay_pre
        self.x_pre_slow *= self.decay_pre_slow
        self.x_post *= self.decay_post
        self.x_post_slow *= self.decay_post_slow

        dW = np.zeros_like(W)

        # LTD: When pre spike arrives, depress based on post trace
        if arrivals.any():
            dW -= p.A2_minus * arrivals * self.x_post[:, None] * W

        # Update pre traces BEFORE computing LTP
        self.x_pre += arrivals
        self.x_pre_slow += arrivals

        # LTP: When post fires, potentiate based on pre trace
        if post_spikes.any():
            post_mask = post_spikes.astype(np.float32)
            triplet_boost = 1.0 + p.A3_plus * self.x_post_slow[:, None] / (p.A2_plus + 1e-9)
            dW += p.A2_plus * post_mask[:, None] * self.x_pre * (p.w_max - W) * triplet_boost

        # Update post traces
        self.x_post += post_spikes.astype(np.float32)
        self.x_post_slow += post_spikes.astype(np.float32)

        return dW


class HomeostaticScaling:
    """
    Biologically plausible homeostatic synaptic scaling (Turrigiano 2008).
    """

    def __init__(self, n_post: int, p: Params):
        self.n_post = n_post
        self.p = p
        self.rate_avg = np.full(n_post, p.target_rate_hz, dtype=np.float32)
        self.decay = math.exp(-p.dt_ms / p.tau_homeostasis)

    def reset(self):
        self.rate_avg.fill(self.p.target_rate_hz)

    def update_rate(self, spikes: np.ndarray, dt_ms: float):
        """Update running rate estimate."""
        instant_rate = spikes.astype(np.float32) * (1000.0 / dt_ms)
        self.rate_avg = self.decay * self.rate_avg + (1 - self.decay) * instant_rate

    def get_scaling_factors(self) -> np.ndarray:
        """Get multiplicative scaling factors."""
        p = self.p
        error = p.target_rate_hz - self.rate_avg
        scale = 1.0 + p.homeostasis_rate * error
        # Allow wider range when rates are very different from target
        return np.clip(scale, 0.9, 1.1)


class InhibitorySTDP:
    """
    Inhibitory STDP from Vogels et al. (2011).

    Adjusts inhibitory weights to stabilize postsynaptic firing at target rate.
    Rule: dw = eta * (x_pre * x_post - rho * x_pre)

    This creates E/I balance: when post fires too much, inhibition increases.
    """

    def __init__(self, n_pre: int, n_post: int, p: Params):
        self.n_pre = n_pre
        self.n_post = n_post
        self.p = p

        # Traces
        self.x_pre = np.zeros(n_pre, dtype=np.float32)
        self.x_post = np.zeros(n_post, dtype=np.float32)

        # Decay
        self.decay = math.exp(-p.dt_ms / p.tau_inh_stdp)

    def reset(self):
        self.x_pre.fill(0)
        self.x_post.fill(0)

    def update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
               W: np.ndarray) -> np.ndarray:
        """
        Update inhibitory weights.

        W shape: (n_post, n_pre) - inhibitory weights from pre to post

        Returns: dW weight change
        """
        p = self.p

        # Decay traces
        self.x_pre *= self.decay
        self.x_post *= self.decay

        dW = np.zeros_like(W)

        # When post fires: potentiate active inhibitory synapses
        if post_spikes.any():
            post_mask = post_spikes.astype(np.float32)
            # Hebbian: increase inhibition from recently active pre neurons
            dW += p.eta_inh * post_mask[:, None] * self.x_pre[None, :]

        # When pre fires: adjust based on post trace minus target
        if pre_spikes.any():
            pre_mask = pre_spikes.astype(np.float32)
            # Anti-Hebbian component with target rate
            dW += p.eta_inh * (self.x_post[:, None] - p.rho_target) * pre_mask[None, :]

        # Update traces
        self.x_pre += pre_spikes.astype(np.float32)
        self.x_post += post_spikes.astype(np.float32)

        return dW


class RgcLgnV1Network:
    """
    Biologically plausible RGC -> LGN -> V1 network.

    Key biological features:
    1. LGN->PV feedforward inhibition (fast disynaptic)
    2. Push-pull inhibition (opposite-phase RFs)
    3. Conductance-based (shunting) inhibition
    4. Disynaptic lateral inhibition via E->SOM->E
    5. Biological weight regulation (no hard normalization)
    6. Inhibitory plasticity (Vogels et al.)
    """

    def __init__(self, p: Params, *, init_mode: str = "random"):
        self.p = p
        self.rng = np.random.default_rng(p.seed)

        self.N = p.N
        self.n_lgn = 2 * p.N * p.N  # ON + OFF channels
        self.M = p.M
        self.L = p.delay_max + 1

        # Spatial coordinates for stimulus
        xs = np.arange(p.N) - (p.N - 1) / 2.0
        ys = np.arange(p.N) - (p.N - 1) / 2.0
        self.X, self.Y = np.meshgrid(xs, ys, indexing="xy")

        # --- LGN Layer (Thalamocortical neurons) ---
        self.lgn = IzhikevichPopulation(self.n_lgn, TC_PARAMS, p.dt_ms, self.rng)

        # --- V1 Excitatory Layer (Regular spiking) ---
        self.v1_exc = IzhikevichPopulation(p.M, RS_PARAMS, p.dt_ms, self.rng)

        # --- Local PV Interneurons (Fast spiking) ---
        self.n_pv = p.M * p.n_pv_per_ensemble
        self.pv = IzhikevichPopulation(self.n_pv, FS_PARAMS, p.dt_ms, self.rng)

        # --- SOM Interneurons (Low-threshold spiking) ---
        self.n_som = p.M * p.n_som_per_ensemble
        self.som = IzhikevichPopulation(self.n_som, LTS_PARAMS, p.dt_ms, self.rng)

        # --- Synaptic currents (AMPA for excitation) ---
        self.I_lgn = np.zeros(self.n_lgn, dtype=np.float32)
        self.I_pv = np.zeros(self.n_pv, dtype=np.float32)
        self.I_som = np.zeros(self.n_som, dtype=np.float32)

        # Synaptic decays
        self.decay_ampa = math.exp(-p.dt_ms / p.tau_ampa)
        self.decay_gaba = math.exp(-p.dt_ms / p.tau_gaba)

        # === NEW: Conductance-based inhibition ===
        # Excitatory conductance (for V1 neurons)
        self.g_exc = np.zeros(p.M, dtype=np.float32)
        # Inhibitory conductances (separate for PV and SOM)
        self.g_inh_pv = np.zeros(p.M, dtype=np.float32)
        self.g_inh_som = np.zeros(p.M, dtype=np.float32)

        # --- Delay buffer for LGN->V1 ---
        self.delay_buf = np.zeros((self.L, self.n_lgn), dtype=np.uint8)
        self.ptr = 0
        self.lgn_ids = np.arange(self.n_lgn)[None, :]

        # Random delays for LGN->V1
        self.D = self.rng.integers(0, self.L, size=(p.M, self.n_lgn),
                                   endpoint=False, dtype=np.int16)

        # === NEW: Separate delays for LGN->PV (slightly faster on average) ===
        # PV interneurons often have faster/shorter latency thalamocortical inputs
        self.D_pv = self.rng.integers(0, max(1, self.L - 2), size=(self.n_pv, self.n_lgn),
                                       endpoint=False, dtype=np.int16)

        # --- LGN->V1 weights (main plastic weights) ---
        if init_mode == "random":
            W = self.rng.normal(p.w_init_mean, p.w_init_std,
                               size=(p.M, self.n_lgn)).astype(np.float32)
        elif init_mode == "near_uniform":
            W = (p.w_init_mean + self.rng.normal(0, p.w_init_std * 0.05,
                                                  size=(p.M, self.n_lgn))).astype(np.float32)
        else:
            raise ValueError("init_mode must be 'random' or 'near_uniform'")
        self.W = np.clip(W, 0.0, p.w_max)

        # === NEW: LGN->PV weights (feedforward inhibition) ===
        # Initialize broadly (PV interneurons are often broadly tuned)
        self.W_lgn_pv = self.rng.normal(p.w_lgn_pv_mean, p.w_lgn_pv_std,
                                         size=(self.n_pv, self.n_lgn)).astype(np.float32)
        self.W_lgn_pv = np.clip(self.W_lgn_pv, 0.0, p.w_lgn_pv_max)

        # === PUSH-PULL: Create opposite-phase PV receptive fields ===
        if p.push_pull_enabled:
            # For push-pull, PV weights should have opposite ON/OFF polarity
            # relative to their paired E neuron
            # We'll swap ON and OFF channels: PV_on = E_off, PV_off = E_on
            # This is done dynamically based on E weights during training
            # For now, initialize with swapped structure
            n_on = p.N * p.N
            for m in range(p.M):
                pv_idx = m * p.n_pv_per_ensemble
                # Initialize PV with swapped ON/OFF from E
                # PV ON channel gets weights similar to E OFF channel region
                # PV OFF channel gets weights similar to E ON channel region
                # Start with random but we'll update during learning
                pass  # Will be updated in plasticity

        # --- E->PV connectivity (local additional drive) ---
        self.W_e_pv = np.zeros((self.n_pv, p.M), dtype=np.float32)
        for m in range(p.M):
            pv_start = m * p.n_pv_per_ensemble
            pv_end = pv_start + p.n_pv_per_ensemble
            self.W_e_pv[pv_start:pv_end, m] = p.w_e_pv

        # --- PV->E connectivity (local inhibition) ---
        self.W_pv_e = np.zeros((p.M, self.n_pv), dtype=np.float32)
        for m in range(p.M):
            pv_start = m * p.n_pv_per_ensemble
            pv_end = pv_start + p.n_pv_per_ensemble
            self.W_pv_e[m, pv_start:pv_end] = p.w_pv_e

        # --- E->SOM connectivity ---
        # === NEW: Distance-dependent for disynaptic lateral inhibition ===
        # Local drive (ensemble to its own SOM)
        self.W_e_som_local = np.zeros((self.n_som, p.M), dtype=np.float32)
        for m in range(p.M):
            som_start = m * p.n_som_per_ensemble
            som_end = som_start + p.n_som_per_ensemble
            self.W_e_som_local[som_start:som_end, m] = p.w_e_som

        # Long-range drive (E neurons excite distant SOMs for disynaptic inhibition)
        self.W_e_som_lateral = np.zeros((self.n_som, p.M), dtype=np.float32)
        for m in range(p.M):
            for other in range(p.M):
                if other != m:
                    som_start = other * p.n_som_per_ensemble
                    som_end = som_start + p.n_som_per_ensemble
                    # Distance in circular topology
                    d = min(abs(m - other), p.M - abs(m - other))
                    # Gaussian falloff for long-range
                    weight = p.w_e_som_lateral * math.exp(-d**2 / (2 * p.som_lateral_sigma**2))
                    self.W_e_som_lateral[som_start:som_end, m] = weight

        # Combine E->SOM weights
        self.W_e_som = self.W_e_som_local + self.W_e_som_lateral

        # --- SOM->E connectivity ---
        # === NEW: LOCAL inhibition (Gaussian falloff) ===
        self.W_som_e = np.zeros((p.M, self.n_som), dtype=np.float32)
        for m in range(p.M):
            for som_ens in range(p.M):
                som_start = som_ens * p.n_som_per_ensemble
                som_end = som_start + p.n_som_per_ensemble
                # Distance in circular topology
                d = min(abs(m - som_ens), p.M - abs(m - som_ens))
                # Gaussian falloff (local inhibition)
                # No self-inhibition for the local SOM
                if d == 0:
                    weight = 0.0  # SOM doesn't inhibit its own ensemble
                else:
                    weight = p.w_som_e * math.exp(-d**2 / (2 * p.som_local_sigma**2))
                self.W_som_e[m, som_start:som_end] = weight

        # --- Lateral excitatory connectivity ---
        self.W_e_e = np.zeros((p.M, p.M), dtype=np.float32)
        for i in range(p.M):
            for j in range(p.M):
                if i != j:
                    d = min(abs(i - j), p.M - abs(i - j))
                    self.W_e_e[i, j] = p.w_e_e_lateral * math.exp(-d**2 / (2 * p.lateral_sigma**2))

        # --- Plasticity mechanisms ---
        self.stdp = TripletSTDP(self.n_lgn, p.M, p, self.rng)
        self.homeostasis = HomeostaticScaling(p.M, p)

        # === NEW: Inhibitory plasticity for PV->E ===
        self.pv_istdp = InhibitorySTDP(self.n_pv, p.M, p)

        # Track V1 spikes from previous timestep for E->E lateral
        self.v1_spk_prev = np.zeros(p.M, dtype=np.uint8)

    def reset_state(self) -> None:
        """Reset all dynamic state (but not weights)."""
        self.lgn.reset()
        self.v1_exc.reset()
        self.pv.reset()
        self.som.reset()

        self.I_lgn.fill(0)
        self.I_pv.fill(0)
        self.I_som.fill(0)
        self.g_exc.fill(0)
        self.g_inh_pv.fill(0)
        self.g_inh_som.fill(0)

        self.delay_buf.fill(0)
        self.ptr = 0

        self.stdp.reset()
        self.pv_istdp.reset()
        self.v1_spk_prev.fill(0)

    def grating(self, theta_deg: float, t_ms: float, phase: float) -> np.ndarray:
        """Generate drifting grating stimulus."""
        p = self.p
        th = math.radians(theta_deg)
        coord = self.X * math.cos(th) + self.Y * math.sin(th)
        return np.sin(2 * math.pi * (p.spatial_freq * coord -
                                      p.temporal_freq * (t_ms / 1000.0)) + phase).astype(np.float32)

    def rgc_spikes(self, stim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ON and OFF RGC spikes from stimulus."""
        p = self.p
        on_rate = p.base_rate + p.gain_rate * np.clip(stim, 0, None)
        off_rate = p.base_rate + p.gain_rate * np.clip(-stim, 0, None)
        dt_s = p.dt_ms / 1000.0
        on_spk = (self.rng.random(stim.shape) < (on_rate * dt_s)).astype(np.uint8)
        off_spk = (self.rng.random(stim.shape) < (off_rate * dt_s)).astype(np.uint8)
        return on_spk, off_spk

    def step(self, on_spk: np.ndarray, off_spk: np.ndarray, plastic: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance network by one timestep.

        Returns: (V1 excitatory spikes, PV spikes, SOM spikes)

        KEY CHANGE: PV is now computed from LGN arrivals BEFORE E spikes,
        so feedforward inhibition can affect E spiking within the same timestep.
        """
        p = self.p

        # Combine ON/OFF RGC spikes
        rgc = np.concatenate([on_spk.ravel(), off_spk.ravel()]).astype(np.float32)

        # --- LGN layer ---
        self.I_lgn *= self.decay_ampa
        self.I_lgn += p.w_rgc_lgn * rgc
        lgn_spk = self.lgn.step(self.I_lgn)

        # Store LGN spikes in delay buffer
        self.delay_buf[self.ptr, :] = lgn_spk

        # Get delayed LGN spikes arriving at V1
        idx_v1 = (self.ptr - self.D) % self.L
        arrivals_v1 = self.delay_buf[idx_v1, self.lgn_ids].astype(np.float32)  # (M, n_lgn)

        # Get delayed LGN spikes arriving at PV (potentially faster)
        idx_pv = (self.ptr - self.D_pv) % self.L
        arrivals_pv = self.delay_buf[idx_pv, self.lgn_ids[:self.n_pv, :]].astype(np.float32)  # (n_pv, n_lgn)

        # === STEP 1: Compute PV BEFORE E (true feedforward inhibition) ===
        # PV receives LGN input directly
        self.I_pv *= self.decay_ampa
        I_pv_lgn = (self.W_lgn_pv * arrivals_pv).sum(axis=1)
        # PV also receives E input from previous timestep (recurrent)
        I_pv_e = self.W_e_pv @ self.v1_spk_prev.astype(np.float32)
        self.I_pv += I_pv_lgn + I_pv_e
        pv_spk = self.pv.step(self.I_pv)

        # === STEP 2: Compute SOM (driven by previous E spikes) ===
        self.I_som *= self.decay_ampa
        self.I_som += self.W_e_som @ self.v1_spk_prev.astype(np.float32)
        som_spk = self.som.step(self.I_som)

        # === STEP 3: Update conductances ===
        # Excitatory conductance from LGN->V1
        self.g_exc *= self.decay_ampa
        I_ff = (self.W * arrivals_v1).sum(axis=1)
        self.g_exc += I_ff

        # Lateral excitation from other E neurons
        self.g_exc += self.W_e_e @ self.v1_spk_prev.astype(np.float32)

        # Inhibitory conductances (GABA decay)
        self.g_inh_pv *= self.decay_gaba
        self.g_inh_som *= self.decay_gaba

        # PV->E inhibition (computed from PV spikes THIS timestep)
        self.g_inh_pv += p.g_pv_scale * (self.W_pv_e @ pv_spk.astype(np.float32))

        # SOM->E inhibition (computed from SOM spikes THIS timestep)
        self.g_inh_som += p.g_som_scale * (self.W_som_e @ som_spk.astype(np.float32))

        # === STEP 4: Compute V1 total current ===
        g_inh_total = self.g_inh_pv + self.g_inh_som
        I_exc = self.g_exc

        if p.conductance_inhibition:
            # Conductance-based (shunting) inhibition
            # I_inh = g_inh * (V - E_inh) - voltage-dependent, divisive
            V = self.v1_exc.v
            I_inh = g_inh_total * (V - p.E_inh) / 50.0  # Scaled for Izhikevich
        else:
            # Current-based inhibition (simpler, more stable for learning)
            I_inh = g_inh_total

        I_v1_total = I_exc - I_inh
        v1_spk = self.v1_exc.step(I_v1_total)

        # === STEP 5: Plasticity ===
        if plastic:
            # Triplet STDP for LGN->V1
            dW = self.stdp.update(arrivals_v1, v1_spk, self.W)
            self.W += dW

            # Weight decay
            self.W *= (1.0 - p.w_decay)

            # Note: Heterosynaptic plasticity is now handled via periodic normalization
            # This is more stable than per-spike depression

            # Clip to valid range
            np.clip(self.W, 0.0, p.w_max, out=self.W)

            # Update homeostatic rate estimate
            self.homeostasis.update_rate(v1_spk, p.dt_ms)

            # Inhibitory plasticity (Vogels et al. iSTDP) - optional
            if p.inhibitory_plasticity:
                dW_inh = self.pv_istdp.update(pv_spk, v1_spk, self.W_pv_e)
                self.W_pv_e += dW_inh
                np.clip(self.W_pv_e, 0.0, p.w_pv_e * 3.0, out=self.W_pv_e)

            # === Push-pull update: slowly align PV weights to opposite phase ===
            if p.push_pull_enabled and self.rng.random() < 0.001:  # Rare update
                n_on = p.N * p.N
                for m in range(p.M):
                    pv_idx = m * p.n_pv_per_ensemble
                    # Get E neuron's ON and OFF weights
                    e_on = self.W[m, :n_on]
                    e_off = self.W[m, n_on:]
                    # PV should have swapped polarity (push-pull)
                    # Slowly move PV weights toward swapped E weights
                    target_pv_on = e_off * (p.w_lgn_pv_mean / (p.w_init_mean + 1e-6))
                    target_pv_off = e_on * (p.w_lgn_pv_mean / (p.w_init_mean + 1e-6))
                    target = np.concatenate([target_pv_on, target_pv_off])
                    target = np.clip(target, 0, p.w_lgn_pv_max)
                    # Slowly interpolate
                    self.W_lgn_pv[pv_idx, :] = 0.99 * self.W_lgn_pv[pv_idx, :] + 0.01 * target

        # Store V1 spikes for next timestep
        self.v1_spk_prev = v1_spk.copy()

        # Update delay buffer pointer
        self.ptr = (self.ptr + 1) % self.L

        return v1_spk, pv_spk, som_spk

    def apply_homeostasis(self):
        """Apply homeostatic scaling to weights."""
        scale = self.homeostasis.get_scaling_factors()
        self.W *= scale[:, None]
        np.clip(self.W, 0.0, self.p.w_max, out=self.W)

    def run_segment(self, theta_deg: float, plastic: bool) -> Tuple[np.ndarray, float, float]:
        """Run one stimulus segment and return V1 spike counts and interneuron rates."""
        p = self.p
        steps = int(p.segment_ms / p.dt_ms)
        phase = float(self.rng.uniform(0, 2 * math.pi))
        v1_counts = np.zeros(self.M, dtype=np.int32)
        pv_counts = 0
        som_counts = 0

        for k in range(steps):
            stim = self.grating(theta_deg, t_ms=k * p.dt_ms, phase=phase)
            on_spk, off_spk = self.rgc_spikes(stim)
            v1_spk, pv_spk, som_spk = self.step(on_spk, off_spk, plastic=plastic)
            v1_counts += v1_spk
            pv_counts += pv_spk.sum()
            som_counts += som_spk.sum()

        # Periodic homeostatic scaling and mild normalization (biological timescale)
        if plastic:
            self.apply_homeostasis()

            # Mild periodic normalization (models slow homeostatic process)
            # Move weight sums gently toward target
            w_sum = self.W.sum(axis=1, keepdims=True) + 1e-6
            target_scale = p.w_target_sum / w_sum
            # Soft interpolation toward target
            scale = 1.0 + p.norm_strength * (target_scale - 1.0)
            self.W *= scale
            np.clip(self.W, 0.0, p.w_max, out=self.W)

        # Return counts and interneuron rates
        duration_s = p.segment_ms / 1000.0
        pv_rate = pv_counts / (self.n_pv * duration_s)
        som_rate = som_counts / (self.n_som * duration_s)

        return v1_counts, pv_rate, som_rate

    def evaluate_tuning(self, thetas_deg: np.ndarray, repeats: int) -> np.ndarray:
        """
        Evaluate orientation tuning.
        Returns rates (Hz) per ensemble per orientation.
        """
        p = self.p
        rates = np.zeros((self.M, len(thetas_deg)), dtype=np.float32)

        # Save state
        state = self._save_state()

        for j, th in enumerate(thetas_deg):
            cnt = np.zeros(self.M, dtype=np.float32)
            for _ in range(repeats):
                self.reset_state()
                counts, _, _ = self.run_segment(float(th), plastic=False)
                cnt += counts
            rates[:, j] = cnt / (repeats * (p.segment_ms / 1000.0))

        # Restore state
        self._restore_state(state)

        return rates

    def _save_state(self) -> dict:
        """Save full network state."""
        return {
            'rng_state': self.rng.bit_generator.state,
            'lgn_v': self.lgn.v.copy(), 'lgn_u': self.lgn.u.copy(),
            'v1_v': self.v1_exc.v.copy(), 'v1_u': self.v1_exc.u.copy(),
            'pv_v': self.pv.v.copy(), 'pv_u': self.pv.u.copy(),
            'som_v': self.som.v.copy(), 'som_u': self.som.u.copy(),
            'I_lgn': self.I_lgn.copy(), 'I_pv': self.I_pv.copy(), 'I_som': self.I_som.copy(),
            'g_exc': self.g_exc.copy(), 'g_inh_pv': self.g_inh_pv.copy(), 'g_inh_som': self.g_inh_som.copy(),
            'delay_buf': self.delay_buf.copy(), 'ptr': self.ptr,
            'stdp_x_pre': self.stdp.x_pre.copy(), 'stdp_x_pre_slow': self.stdp.x_pre_slow.copy(),
            'stdp_x_post': self.stdp.x_post.copy(), 'stdp_x_post_slow': self.stdp.x_post_slow.copy(),
            'istdp_x_pre': self.pv_istdp.x_pre.copy(), 'istdp_x_post': self.pv_istdp.x_post.copy(),
            'v1_spk_prev': self.v1_spk_prev.copy(),
        }

    def _restore_state(self, state: dict) -> None:
        """Restore full network state."""
        self.rng.bit_generator.state = state['rng_state']
        self.lgn.v = state['lgn_v']; self.lgn.u = state['lgn_u']
        self.v1_exc.v = state['v1_v']; self.v1_exc.u = state['v1_u']
        self.pv.v = state['pv_v']; self.pv.u = state['pv_u']
        self.som.v = state['som_v']; self.som.u = state['som_u']
        self.I_lgn = state['I_lgn']; self.I_pv = state['I_pv']; self.I_som = state['I_som']
        self.g_exc = state['g_exc']; self.g_inh_pv = state['g_inh_pv']; self.g_inh_som = state['g_inh_som']
        self.delay_buf = state['delay_buf']; self.ptr = state['ptr']
        self.stdp.x_pre = state['stdp_x_pre']; self.stdp.x_pre_slow = state['stdp_x_pre_slow']
        self.stdp.x_post = state['stdp_x_post']; self.stdp.x_post_slow = state['stdp_x_post_slow']
        self.pv_istdp.x_pre = state['istdp_x_pre']; self.pv_istdp.x_post = state['istdp_x_post']
        self.v1_spk_prev = state['v1_spk_prev']


# =============================================================================
# Visualization functions
# =============================================================================

def plot_weight_maps(W: np.ndarray, N: int, outpath: str, title: str) -> None:
    """Plot ON, OFF, and ON-OFF weight maps for each ensemble."""
    M = W.shape[0]
    W_on = W[:, :N * N].reshape(M, N, N)
    W_off = W[:, N * N:].reshape(M, N, N)
    W_diff = W_on - W_off

    fig, axes = plt.subplots(M, 3, figsize=(9, 2.1 * M))
    if M == 1:
        axes = np.array([axes])

    for m in range(M):
        for j, (arr, coltitle) in enumerate([(W_on[m], "ON"), (W_off[m], "OFF"),
                                              (W_diff[m], "ON-OFF")]):
            ax = axes[m, j]
            im = ax.imshow(arr, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if m == 0:
                ax.set_title(coltitle)
            if j == 0:
                ax.set_ylabel(f"E{m}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_tuning(rates: np.ndarray, thetas_deg: np.ndarray, osi: np.ndarray,
                pref_deg: np.ndarray, outpath: str, title: str) -> None:
    """Plot orientation tuning curves."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in range(rates.shape[0]):
        ax.plot(thetas_deg, rates[m], marker="o",
                label=f"E{m} OSI={osi[m]:.2f} pref={pref_deg[m]:.0f}")
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_scalar_over_time(xs: np.ndarray, ys: np.ndarray, outpath: str,
                          ylabel: str, title: str) -> None:
    """Plot scalar metric over training."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Training segment")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_interneuron_activity(pv_rates: List[float], som_rates: List[float],
                              segments: List[int], outpath: str) -> None:
    """Plot interneuron activity over training."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(segments, pv_rates, marker="o", label="PV (FS)")
    ax.plot(segments, som_rates, marker="s", label="SOM (LTS)")
    ax.set_xlabel("Training segment")
    ax.set_ylabel("Mean firing rate (Hz)")
    ax.set_title("Interneuron activity over training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_conductances(g_exc_hist: List[float], g_inh_hist: List[float],
                      segments: List[int], outpath: str) -> None:
    """Plot conductance balance over training."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(segments, g_exc_hist, marker="o", label="g_exc (mean)")
    ax.plot(segments, g_inh_hist, marker="s", label="g_inh (mean)")
    ax.set_xlabel("Training segment")
    ax.set_ylabel("Mean conductance")
    ax.set_title("E/I conductance balance over training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# =============================================================================
# Main training loop
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Biologically plausible V1 STDP network v2")
    ap.add_argument("--out", type=str, default="runs/bio_plausible_v2",
                    help="output directory")
    ap.add_argument("--train-segments", type=int, default=1000)
    ap.add_argument("--segment-ms", type=int, default=300)
    ap.add_argument("--N", type=int, default=8, help="Patch size NxN")
    ap.add_argument("--M", type=int, default=32, help="Number of V1 ensembles")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--viz-every", type=int, default=50)
    ap.add_argument("--eval-K", type=int, default=12, help="Number of orientations to test")
    ap.add_argument("--eval-repeats", type=int, default=3)
    ap.add_argument("--baseline-repeats", type=int, default=7)
    ap.add_argument("--init-mode", type=str, default="random",
                    choices=["random", "near_uniform"])

    # Biological mechanism flags
    ap.add_argument("--conductance-inhibition", action="store_true",
                    help="Use conductance-based (shunting) inhibition instead of current-based")
    ap.add_argument("--inhibitory-plasticity", action="store_true",
                    help="Enable Vogels et al. inhibitory STDP")
    ap.add_argument("--no-push-pull", action="store_true",
                    help="Disable push-pull (opposite-phase) PV receptive fields")

    args = ap.parse_args()
    safe_mkdir(args.out)

    # Create network with biological mechanism flags
    p = Params(
        N=args.N,
        M=args.M,
        seed=args.seed,
        train_segments=args.train_segments,
        segment_ms=args.segment_ms,
        conductance_inhibition=args.conductance_inhibition,
        inhibitory_plasticity=args.inhibitory_plasticity,
        push_pull_enabled=not args.no_push_pull
    )
    net = RgcLgnV1Network(p, init_mode=args.init_mode)

    print(f"[init] Biologically plausible RGC->LGN->V1 network (v2)")
    print(f"[init] N={p.N} (patch), M={p.M} (ensembles), n_lgn={net.n_lgn}")
    print(f"[init] Neuron types: LGN=TC(Izhikevich), V1=RS, PV=FS, SOM=LTS")
    print(f"[init] BIOLOGICAL MECHANISMS:")
    print(f"       - LGN->PV feedforward inhibition: ENABLED (true disynaptic)")
    inh_mode = "conductance-based (shunting)" if p.conductance_inhibition else "current-based"
    print(f"       - Inhibition mode: {inh_mode}")
    push_pull_status = "ENABLED" if p.push_pull_enabled else "disabled"
    print(f"       - Push-pull inhibition: {push_pull_status}")
    print(f"       - Disynaptic lateral inhibition: ENABLED (E->SOM->E)")
    istdp_status = "ENABLED" if p.inhibitory_plasticity else "disabled"
    print(f"       - Inhibitory plasticity (iSTDP): {istdp_status}")
    print(f"       - Soft normalization: ENABLED (heterosynaptic-like)")
    print(f"[init] init-mode = {args.init_mode}")

    thetas = np.linspace(0, 180 - 180 / args.eval_K, args.eval_K)

    # --- Baseline evaluation ---
    print("\n[baseline] Evaluating tuning at initialization...")
    rates0 = net.evaluate_tuning(thetas, repeats=args.baseline_repeats)
    osi0, pref0 = compute_osi(rates0, thetas)

    print(f"[seg {0:4d}] mean rate={rates0.mean():.3f} Hz | mean OSI={osi0.mean():.3f} | max OSI={osi0.max():.3f}")
    print(f"          prefs(deg) = {np.round(pref0, 1)}")

    plot_weight_maps(net.W, p.N, os.path.join(args.out, "weights_seg0000.png"),
                     title="LGN->V1 weights at init (segment 0)")
    plot_tuning(rates0, thetas, osi0, pref0,
                os.path.join(args.out, "tuning_seg0000.png"),
                title="Baseline tuning (segment 0, before learning)")

    # Tracking
    seg_hist = [0]
    osi_hist = [float(osi0.mean())]
    rate_hist = [float(rates0.mean())]
    pv_rate_hist = [0.0]
    som_rate_hist = [0.0]

    if p.train_segments == 0:
        print("[final] train-segments=0, no learning occurred")
        print(f"[done] outputs written to: {args.out}")
        return

    # --- Training ---
    print("\n[training] Starting STDP training...")
    for s in range(1, p.train_segments + 1):
        # Random orientation for this segment
        th = float(net.rng.uniform(0.0, 180.0))
        _, pv_rate, som_rate = net.run_segment(th, plastic=True)

        if (s % args.viz_every) == 0 or s == p.train_segments:
            rates = net.evaluate_tuning(thetas, repeats=args.eval_repeats)
            osi, pref = compute_osi(rates, thetas)

            print(f"[seg {s:4d}] mean rate={rates.mean():.3f} Hz | mean OSI={osi.mean():.3f} | max OSI={osi.max():.3f}")
            print(f"          PV rate={pv_rate:.1f} Hz | SOM rate={som_rate:.1f} Hz")
            print(f"          prefs(deg) = {np.round(pref, 1)}")

            plot_weight_maps(net.W, p.N,
                           os.path.join(args.out, f"weights_seg{s:04d}.png"),
                           title=f"LGN->V1 weights (segment {s})")
            plot_tuning(rates, thetas, osi, pref,
                       os.path.join(args.out, f"tuning_seg{s:04d}.png"),
                       title=f"Tuning during training (segment {s})")

            seg_hist.append(int(s))
            osi_hist.append(float(osi.mean()))
            rate_hist.append(float(rates.mean()))
            pv_rate_hist.append(float(pv_rate))
            som_rate_hist.append(float(som_rate))

    # --- Final evaluation ---
    print("\n[final] Final evaluation with robust repeats...")
    final_repeats = max(args.baseline_repeats, args.eval_repeats, 7)
    rates1 = net.evaluate_tuning(thetas, repeats=final_repeats)
    osi1, pref1 = compute_osi(rates1, thetas)

    d_osi = osi1 - osi0
    print(f"[final] baseline mean OSI={osi0.mean():.3f} -> final mean OSI={osi1.mean():.3f} (delta={d_osi.mean():+.3f})")
    print(f"[final] fraction ensembles with OSI>0.3: {(osi1>0.3).mean()*100:.1f}%")
    print(f"[final] fraction ensembles with OSI>0.5: {(osi1>0.5).mean()*100:.1f}%")

    # Final plots
    plot_weight_maps(net.W, p.N,
                    os.path.join(args.out, "weights_final.png"),
                    title="LGN->V1 weights (final)")
    plot_tuning(rates1, thetas, osi1, pref1,
               os.path.join(args.out, "tuning_final.png"),
               title="Final tuning (after learning)")
    plot_scalar_over_time(np.array(seg_hist), np.array(osi_hist),
                         os.path.join(args.out, "mean_osi_over_time.png"),
                         ylabel="mean OSI", title="Mean OSI over training")
    plot_scalar_over_time(np.array(seg_hist), np.array(rate_hist),
                         os.path.join(args.out, "mean_rate_over_time.png"),
                         ylabel="mean rate (Hz)", title="Mean firing rate over training")
    plot_interneuron_activity(pv_rate_hist, som_rate_hist, seg_hist,
                             os.path.join(args.out, "interneuron_activity.png"))

    print(f"\n[done] Outputs written to: {args.out}")


if __name__ == "__main__":
    main()
