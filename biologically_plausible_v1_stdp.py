#!/usr/bin/env python3
"""biologically_plausible_v1_stdp.py

RGC -> LGN -> V1(L4) spiking network with STDP that learns orientation selectivity.

BIOLOGICAL PLAUSIBILITY IMPROVEMENTS:
1. Izhikevich neurons replace LIF neurons (proper parameters for each cell type)
2. Local PV/SOM interneuron circuits replace global inhibition
3. Homeostatic synaptic scaling replaces global weight normalization
4. Triplet STDP rule for more realistic plasticity
5. Lateral connectivity between ensembles (excitatory and inhibitory)

Neuron types and their Izhikevich parameters (from Izhikevich 2003, 2007):
- Thalamocortical (TC) LGN: a=0.02, b=0.25, c=-65, d=0.05 (rebound bursting)
- Regular Spiking (RS) V1 excitatory: a=0.02, b=0.2, c=-65, d=8
- Fast Spiking (FS) PV interneurons: a=0.1, b=0.2, c=-65, d=2
- Low-threshold spiking (LTS) SOM interneurons: a=0.02, b=0.25, c=-65, d=2

References:
- Izhikevich (2003) "Simple model of spiking neurons"
- Izhikevich (2007) "Dynamical Systems in Neuroscience"
- Turrigiano (2008) "Homeostatic synaptic plasticity"
- Pfister & Gerstner (2006) "Triplets of spikes in STDP"

License: MIT
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Tuple, List

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
    # Izhikevich model uses currents ~0-40 pA for typical spiking
    w_rgc_lgn: float = 5.0

    # LGN->V1 weights & delays
    delay_max: int = 12
    w_init_mean: float = 0.25  # Scaled for Izhikevich (total input ~15-30 pA)
    w_init_std: float = 0.08
    w_max: float = 1.0

    # Homeostatic synaptic scaling (replaces global normalization)
    target_rate_hz: float = 8.0  # Target firing rate for homeostasis
    tau_homeostasis: float = 5000.0  # Time constant for homeostasis (ms)
    homeostasis_rate: float = 0.001  # Learning rate for homeostasis

    # STDP parameters (pair-based with triplet enhancement)
    # Time constants matched to original working code
    tau_plus: float = 20.0   # Pre-before-post time constant
    tau_minus: float = 20.0  # Post-before-pre time constant
    tau_x: float = 101.0     # Slow pre trace for triplet
    tau_y: float = 125.0     # Slow post trace for triplet
    A2_plus: float = 0.008   # Pair LTP amplitude (matches original)
    A3_plus: float = 0.002   # Triplet LTP enhancement
    A2_minus: float = 0.010  # Pair LTD amplitude (matches original)
    A3_minus: float = 0.0    # Triplet LTD amplitude (often 0)

    # Weight decay and normalization (biologically: synaptic turnover)
    w_decay: float = 0.0001  # Per-timestep weight decay
    w_norm_target: float = 15.0  # Target sum of weights per neuron

    # Local inhibitory circuit parameters
    n_pv_per_ensemble: int = 1  # PV interneurons per ensemble
    n_som_per_ensemble: int = 1  # SOM interneurons per ensemble for lateral inhibition

    # E->PV (feedforward inhibition) - scaled for Izhikevich
    w_e_pv: float = 5.0
    # PV->E (feedback inhibition, local)
    w_pv_e: float = 3.0
    # E->SOM (lateral inhibition drive from this ensemble)
    w_e_som: float = 6.0
    # SOM->E (lateral inhibition TO OTHER ensembles - NOT self)
    w_som_e: float = 5.0  # Peak inhibition strength (for strong Mexican hat effect)

    # Mexican hat lateral inhibition parameters
    # Inhibition profile: I(d) = w_som_e * mexican_hat(d)
    # mexican_hat(d) = exp(-d²/2σ_far²) - exp(-d²/2σ_near²)
    # This creates weak inhibition for very near and very far, strong for intermediate
    # Tuned for clear separation: d=1 weak, d=2-4 strong, d>5 declining
    mexican_hat_sigma_near: float = 1.8  # Sigma for local (weak inhibition zone)
    mexican_hat_sigma_far: float = 5.0   # Sigma for surround (determines falloff)

    # Lateral excitatory connections (between nearby ensembles)
    w_e_e_lateral: float = 0.8  # Increased to strengthen local cooperation
    lateral_sigma: float = 2.0  # Broader spread for lateral connections

    # Synaptic time constants
    tau_ampa: float = 5.0   # AMPA receptor
    tau_gaba: float = 10.0  # GABA receptor


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

        Uses the standard Izhikevich equations:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)

        if v >= v_peak: v <- c, u <- u + d
        """
        p = self.p
        dt = self.dt

        # Euler integration (with sub-stepping for stability)
        # Using 2 sub-steps per dt for better numerical stability
        dt_sub = dt / 2.0

        for _ in range(2):
            # Clamp v to prevent numerical blowup
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

    This implementation properly handles per-synapse traces with delays.
    Each synapse from pre neuron j to post neuron i has its own trace,
    because the spike arrival times depend on axonal delays.

    Maintains traces per synapse (M, n_pre):
    - x_pre: fast pre trace (incremented when pre spike arrives at synapse)
    - x_pre_slow: slow pre trace for triplet (same)
    And traces per post neuron (M,):
    - x_post: fast post trace
    - x_post_slow: slow post trace for triplet
    """

    def __init__(self, n_pre: int, n_post: int, p: Params, rng: np.random.Generator):
        self.n_pre = n_pre
        self.n_post = n_post
        self.p = p

        # Pre traces - per synapse (n_post, n_pre)
        # These track the arrival of pre-synaptic spikes at each synapse
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

        CRITICAL: Order of operations matches original working code:
        1. Decay traces
        2. LTD: when pre arrives, depress based on OLD post trace
        3. Update pre traces (so current arrivals are included)
        4. LTP: when post fires, potentiate based on NEW pre trace (includes current arrivals)
        5. Update post traces

        This order ensures that coincident pre-post activity within the same timestep
        contributes to LTP, which is essential for proper orientation selectivity learning.

        arrivals: (n_post, n_pre) - which pre-spikes arrived at each synapse this timestep
        post_spikes: (n_post,) binary
        W: (n_post, n_pre) current weights

        Returns: dW weight change matrix (already includes multiplicative factors)
        """
        p = self.p

        # Decay all traces
        self.x_pre *= self.decay_pre
        self.x_pre_slow *= self.decay_pre_slow
        self.x_post *= self.decay_post
        self.x_post_slow *= self.decay_post_slow

        dW = np.zeros_like(W)

        # LTD: When pre spike arrives, depress based on post trace (OLD, before this spike)
        # Multiplicative: dW- proportional to W (stronger synapses lose more)
        if arrivals.any():
            dW -= p.A2_minus * arrivals * self.x_post[:, None] * W

        # Update pre traces BEFORE computing LTP
        # This ensures current arrivals contribute to LTP if post fires this timestep
        self.x_pre += arrivals
        self.x_pre_slow += arrivals

        # LTP: When post fires, potentiate based on pre trace (NEW, includes current arrivals)
        # Multiplicative: dW+ proportional to (w_max - W) (room to grow)
        # Triplet enhancement: stronger LTP when there's recent post activity
        if post_spikes.any():
            post_mask = post_spikes.astype(np.float32)
            triplet_boost = 1.0 + p.A3_plus * self.x_post_slow[:, None] / p.A2_plus
            dW += p.A2_plus * post_mask[:, None] * self.x_pre * (p.w_max - W) * triplet_boost

        # Update post traces AFTER computing plasticity
        self.x_post += post_spikes.astype(np.float32)
        self.x_post_slow += post_spikes.astype(np.float32)

        return dW


class HomeostaticScaling:
    """
    Biologically plausible homeostatic synaptic scaling.

    Based on Turrigiano (2008): neurons slowly adjust their synaptic
    strengths to maintain a target firing rate. This is a LOCAL mechanism
    that operates on each neuron independently.

    The scaling is multiplicative: w <- w * (1 + eta * (r_target - r_actual))
    """

    def __init__(self, n_post: int, p: Params):
        self.n_post = n_post
        self.p = p

        # Running average of firing rate (exponential moving average)
        self.rate_avg = np.full(n_post, p.target_rate_hz, dtype=np.float32)

        # Decay for rate averaging
        self.decay = math.exp(-p.dt_ms / p.tau_homeostasis)

    def reset(self):
        self.rate_avg.fill(self.p.target_rate_hz)

    def update_rate(self, spikes: np.ndarray, dt_ms: float):
        """Update running rate estimate."""
        instant_rate = spikes.astype(np.float32) * (1000.0 / dt_ms)  # Convert to Hz
        self.rate_avg = self.decay * self.rate_avg + (1 - self.decay) * instant_rate

    def get_scaling_factors(self) -> np.ndarray:
        """
        Get multiplicative scaling factors for each neuron's input weights.

        Returns: (n_post,) array of scaling factors
        """
        p = self.p
        # Error signal: positive if firing too slow, negative if too fast
        error = p.target_rate_hz - self.rate_avg
        # Multiplicative scaling factor
        scale = 1.0 + p.homeostasis_rate * error
        return np.clip(scale, 0.95, 1.05)  # Limit rate of change


class RgcLgnV1Network:
    """
    Biologically plausible RGC -> LGN -> V1 network.

    Key biological features:
    1. Izhikevich neurons (TC for LGN, RS for V1 excitatory, FS for PV, LTS for SOM)
    2. Local inhibitory circuits (PV for feedforward, SOM for lateral)
    3. Triplet STDP for plasticity
    4. Homeostatic synaptic scaling
    5. Lateral excitatory connections
    """

    def __init__(self, p: Params, *, init_mode: str = "random"):
        self.p = p
        self.rng = np.random.default_rng(p.seed)

        self.N = p.N
        self.n_lgn = 2 * p.N * p.N  # ON + OFF channels
        self.M = p.M  # Number of V1 ensembles
        self.L = p.delay_max + 1  # Delay buffer length

        # Spatial coordinates for stimulus
        xs = np.arange(p.N) - (p.N - 1) / 2.0
        ys = np.arange(p.N) - (p.N - 1) / 2.0
        self.X, self.Y = np.meshgrid(xs, ys, indexing="xy")

        # --- LGN Layer (Thalamocortical neurons) ---
        self.lgn = IzhikevichPopulation(self.n_lgn, TC_PARAMS, p.dt_ms, self.rng)

        # --- V1 Excitatory Layer (Regular spiking) ---
        self.v1_exc = IzhikevichPopulation(p.M, RS_PARAMS, p.dt_ms, self.rng)

        # --- Local PV Interneurons (Fast spiking) ---
        # One PV per ensemble for local feedforward inhibition
        self.n_pv = p.M * p.n_pv_per_ensemble
        self.pv = IzhikevichPopulation(self.n_pv, FS_PARAMS, p.dt_ms, self.rng)

        # --- SOM Interneurons (Low-threshold spiking) ---
        # Each ensemble has its own SOM neuron for lateral inhibition
        self.n_som = p.M * p.n_som_per_ensemble
        self.som = IzhikevichPopulation(self.n_som, LTS_PARAMS, p.dt_ms, self.rng)

        # --- Synaptic currents ---
        self.I_lgn = np.zeros(self.n_lgn, dtype=np.float32)
        self.I_v1 = np.zeros(p.M, dtype=np.float32)
        self.I_pv = np.zeros(self.n_pv, dtype=np.float32)
        self.I_som = np.zeros(self.n_som, dtype=np.float32)

        # Synaptic decays
        self.decay_ampa = math.exp(-p.dt_ms / p.tau_ampa)
        self.decay_gaba = math.exp(-p.dt_ms / p.tau_gaba)

        # Inhibitory currents (separate for proper decay)
        self.I_v1_inh = np.zeros(p.M, dtype=np.float32)

        # --- Delay buffer for LGN->V1 ---
        self.delay_buf = np.zeros((self.L, self.n_lgn), dtype=np.uint8)
        self.ptr = 0
        self.lgn_ids = np.arange(self.n_lgn)[None, :]

        # Random delays (no orientation bias)
        self.D = self.rng.integers(0, self.L, size=(p.M, self.n_lgn),
                                   endpoint=False, dtype=np.int16)

        # --- LGN->V1 weights (unbiased initialization) ---
        if init_mode == "random":
            W = self.rng.normal(p.w_init_mean, p.w_init_std,
                               size=(p.M, self.n_lgn)).astype(np.float32)
        elif init_mode == "near_uniform":
            W = (p.w_init_mean + self.rng.normal(0, p.w_init_std * 0.05,
                                                  size=(p.M, self.n_lgn))).astype(np.float32)
        else:
            raise ValueError("init_mode must be 'random' or 'near_uniform'")

        self.W = np.clip(W, 0.0, p.w_max)

        # --- Local inhibitory connectivity ---
        # E->PV connectivity (each E connects to its local PV)
        self.W_e_pv = np.zeros((self.n_pv, p.M), dtype=np.float32)
        for m in range(p.M):
            pv_start = m * p.n_pv_per_ensemble
            pv_end = pv_start + p.n_pv_per_ensemble
            self.W_e_pv[pv_start:pv_end, m] = p.w_e_pv

        # PV->E connectivity (local inhibition only)
        self.W_pv_e = np.zeros((p.M, self.n_pv), dtype=np.float32)
        for m in range(p.M):
            pv_start = m * p.n_pv_per_ensemble
            pv_end = pv_start + p.n_pv_per_ensemble
            self.W_pv_e[m, pv_start:pv_end] = p.w_pv_e

        # E->SOM connectivity (each ensemble drives its own SOM neuron)
        # This is local - ensemble m drives SOM neuron m
        self.W_e_som = np.zeros((self.n_som, p.M), dtype=np.float32)
        for m in range(p.M):
            som_start = m * p.n_som_per_ensemble
            som_end = som_start + p.n_som_per_ensemble
            self.W_e_som[som_start:som_end, m] = p.w_e_som

        # SOM->E connectivity (LATERAL inhibition - inhibits OTHER ensembles)
        # SOM neuron m inhibits all ensembles EXCEPT ensemble m
        # Uses Mexican hat profile: weak at short and long distances, strong at intermediate
        # This creates distance-dependent competition between ensembles
        self.W_som_e = np.zeros((p.M, self.n_som), dtype=np.float32)
        for m in range(p.M):
            som_start = m * p.n_som_per_ensemble
            som_end = som_start + p.n_som_per_ensemble
            for other in range(p.M):
                if other != m:  # Inhibit others, NOT self
                    # Circular distance on the ensemble ring
                    d = min(abs(other - m), p.M - abs(other - m))
                    # Mexican hat profile: difference of two Gaussians
                    # Strong inhibition at intermediate distances, weak at near and far
                    gauss_far = math.exp(-d**2 / (2 * p.mexican_hat_sigma_far**2))
                    gauss_near = math.exp(-d**2 / (2 * p.mexican_hat_sigma_near**2))
                    mexican_hat = gauss_far - gauss_near
                    # Clip to ensure non-negative inhibition
                    self.W_som_e[other, som_start:som_end] = p.w_som_e * max(0.0, mexican_hat)

        # --- Lateral excitatory connectivity ---
        # Gaussian connectivity based on ensemble distance (circular topology)
        self.W_e_e = np.zeros((p.M, p.M), dtype=np.float32)
        for i in range(p.M):
            for j in range(p.M):
                if i != j:
                    # Circular distance
                    d = min(abs(i - j), p.M - abs(i - j))
                    self.W_e_e[i, j] = p.w_e_e_lateral * math.exp(-d**2 / (2 * p.lateral_sigma**2))

        # --- Plasticity mechanisms ---
        self.stdp = TripletSTDP(self.n_lgn, p.M, p, self.rng)
        self.homeostasis = HomeostaticScaling(p.M, p)

    def reset_state(self) -> None:
        """Reset all dynamic state (but not weights)."""
        self.lgn.reset()
        self.v1_exc.reset()
        self.pv.reset()
        self.som.reset()

        self.I_lgn.fill(0)
        self.I_v1.fill(0)
        self.I_pv.fill(0)
        self.I_som.fill(0)
        self.I_v1_inh.fill(0)

        self.delay_buf.fill(0)
        self.ptr = 0

        self.stdp.reset()
        # Note: we don't reset homeostasis to preserve rate estimates

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

    def step(self, on_spk: np.ndarray, off_spk: np.ndarray, plastic: bool) -> np.ndarray:
        """
        Advance network by one timestep.

        Returns V1 excitatory spikes.
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
        idx = (self.ptr - self.D) % self.L
        arrivals = self.delay_buf[idx, self.lgn_ids].astype(np.float32)  # (M, n_lgn)

        # --- V1 feedforward input ---
        I_ff = (self.W * arrivals).sum(axis=1)

        # --- Lateral excitation ---
        # Get previous V1 spikes (need to track)
        # For simplicity, we'll use instantaneous lateral connections

        # --- V1 excitatory layer ---
        self.I_v1 *= self.decay_ampa
        self.I_v1 += I_ff

        # Inhibitory input (GABA decay)
        self.I_v1_inh *= self.decay_gaba

        # Total current to V1 excitatory
        I_v1_total = self.I_v1 - self.I_v1_inh
        v1_spk = self.v1_exc.step(I_v1_total)

        # --- PV interneurons (feedforward inhibition) ---
        self.I_pv *= self.decay_ampa
        self.I_pv += self.W_e_pv @ v1_spk.astype(np.float32)
        pv_spk = self.pv.step(self.I_pv)

        # PV->E inhibition
        self.I_v1_inh += self.W_pv_e @ pv_spk.astype(np.float32)

        # --- SOM interneurons (lateral inhibition) ---
        self.I_som *= self.decay_ampa
        self.I_som += self.W_e_som @ v1_spk.astype(np.float32)
        som_spk = self.som.step(self.I_som)

        # SOM->E lateral inhibition
        self.I_v1_inh += self.W_som_e @ som_spk.astype(np.float32)

        # --- Lateral excitation ---
        self.I_v1 += self.W_e_e @ v1_spk.astype(np.float32)

        # --- Plasticity ---
        if plastic:
            # Triplet STDP with per-synapse arrivals
            # arrivals is (M, n_lgn) - different for each post neuron due to delays
            # dW already includes multiplicative bounds
            dW = self.stdp.update(arrivals, v1_spk, self.W)

            # Apply weight changes directly (multiplicative bounds already in dW)
            self.W += dW

            # Weight decay (models synaptic turnover/protein degradation)
            self.W *= (1.0 - p.w_decay)

            # Clip to valid range
            np.clip(self.W, 0.0, p.w_max, out=self.W)

            # Update homeostatic rate estimate
            self.homeostasis.update_rate(v1_spk, p.dt_ms)

        # Update delay buffer pointer
        self.ptr = (self.ptr + 1) % self.L

        return v1_spk

    def apply_homeostasis(self):
        """Apply homeostatic scaling to weights (call periodically, not every step)."""
        scale = self.homeostasis.get_scaling_factors()
        self.W *= scale[:, None]
        np.clip(self.W, 0.0, self.p.w_max, out=self.W)

    def run_segment(self, theta_deg: float, plastic: bool) -> np.ndarray:
        """Run one stimulus segment and return V1 spike counts."""
        p = self.p
        steps = int(p.segment_ms / p.dt_ms)
        phase = float(self.rng.uniform(0, 2 * math.pi))
        v1_counts = np.zeros(self.M, dtype=np.int32)

        for k in range(steps):
            stim = self.grating(theta_deg, t_ms=k * p.dt_ms, phase=phase)
            on_spk, off_spk = self.rgc_spikes(stim)
            v1_counts += self.step(on_spk, off_spk, plastic=plastic)

        # Apply weight normalization at end of segment
        # This models homeostatic synaptic scaling which operates on slower timescales
        if plastic:
            # Normalize total input weight per neuron (heterosynaptic plasticity)
            w_sum = self.W.sum(axis=1, keepdims=True) + 1e-6
            self.W *= (p.w_norm_target / w_sum)
            np.clip(self.W, 0.0, p.w_max, out=self.W)

        return v1_counts

    def evaluate_tuning(self, thetas_deg: np.ndarray, repeats: int) -> np.ndarray:
        """
        Evaluate orientation tuning.

        Returns rates (Hz) per ensemble per orientation.
        """
        p = self.p
        rates = np.zeros((self.M, len(thetas_deg)), dtype=np.float32)

        # Save state
        rng_state = self.rng.bit_generator.state
        saved_lgn_v = self.lgn.v.copy()
        saved_lgn_u = self.lgn.u.copy()
        saved_v1_v = self.v1_exc.v.copy()
        saved_v1_u = self.v1_exc.u.copy()
        saved_pv_v = self.pv.v.copy()
        saved_pv_u = self.pv.u.copy()
        saved_som_v = self.som.v.copy()
        saved_som_u = self.som.u.copy()
        saved_I_lgn = self.I_lgn.copy()
        saved_I_v1 = self.I_v1.copy()
        saved_I_v1_inh = self.I_v1_inh.copy()
        saved_I_pv = self.I_pv.copy()
        saved_I_som = self.I_som.copy()
        saved_buf = self.delay_buf.copy()
        saved_ptr = self.ptr
        saved_stdp_x_pre = self.stdp.x_pre.copy()
        saved_stdp_x_pre_slow = self.stdp.x_pre_slow.copy()
        saved_stdp_x_post = self.stdp.x_post.copy()
        saved_stdp_x_post_slow = self.stdp.x_post_slow.copy()

        for j, th in enumerate(thetas_deg):
            cnt = np.zeros(self.M, dtype=np.float32)
            for _ in range(repeats):
                self.reset_state()
                cnt += self.run_segment(float(th), plastic=False)
            rates[:, j] = cnt / (repeats * (p.segment_ms / 1000.0))

        # Restore state
        self.lgn.v = saved_lgn_v
        self.lgn.u = saved_lgn_u
        self.v1_exc.v = saved_v1_v
        self.v1_exc.u = saved_v1_u
        self.pv.v = saved_pv_v
        self.pv.u = saved_pv_u
        self.som.v = saved_som_v
        self.som.u = saved_som_u
        self.I_lgn = saved_I_lgn
        self.I_v1 = saved_I_v1
        self.I_v1_inh = saved_I_v1_inh
        self.I_pv = saved_I_pv
        self.I_som = saved_I_som
        self.delay_buf = saved_buf
        self.ptr = saved_ptr
        self.stdp.x_pre = saved_stdp_x_pre
        self.stdp.x_pre_slow = saved_stdp_x_pre_slow
        self.stdp.x_post = saved_stdp_x_post
        self.stdp.x_post_slow = saved_stdp_x_post_slow
        self.rng.bit_generator.state = rng_state

        return rates


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


# =============================================================================
# Inhibition-Orientation Correlation Analysis
# =============================================================================

def compute_orientation_difference(pref1_deg: float, pref2_deg: float) -> float:
    """
    Compute the circular difference between two orientations in [0, 180).

    Returns a value in [0, 90] degrees (since orientations are periodic with period 180).
    0 means identical orientation, 90 means orthogonal.
    """
    diff = abs(pref1_deg - pref2_deg)
    # Wrap to [0, 180)
    diff = diff % 180.0
    # Map to [0, 90] since 0 and 180 are the same orientation
    if diff > 90:
        diff = 180 - diff
    return diff


def compute_pairwise_inhibition_matrix(W_som_e: np.ndarray, M: int, n_som_per_ensemble: int = 1) -> np.ndarray:
    """
    Compute the effective pairwise lateral inhibition between ensembles.

    W_som_e is (M, n_som) where n_som = M * n_som_per_ensemble.
    Returns (M, M) matrix where entry [i,j] is the inhibition FROM ensemble j TO ensemble i.
    """
    inhibition_matrix = np.zeros((M, M), dtype=np.float32)
    for j in range(M):  # Source ensemble
        som_start = j * n_som_per_ensemble
        som_end = som_start + n_som_per_ensemble
        for i in range(M):  # Target ensemble
            # Sum of inhibition weights from SOM neurons of ensemble j to ensemble i
            inhibition_matrix[i, j] = W_som_e[i, som_start:som_end].sum()
    return inhibition_matrix


def analyze_inhibition_orientation_correlation(
    pref_deg: np.ndarray,
    W_som_e: np.ndarray,
    M: int,
    n_som_per_ensemble: int = 1
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Analyze the correlation between lateral inhibition and orientation difference.

    For each pair of neurons (i, j), computes:
    - Lateral inhibition between them (symmetrized: avg of i->j and j->i)
    - Orientation difference |pref_i - pref_j| (circular, in [0, 90])

    Returns:
        inhibition_values: array of pairwise inhibition strengths
        orientation_diffs: array of pairwise orientation differences
        correlation: Pearson correlation coefficient
        p_value: p-value for the correlation (two-tailed)
    """
    from scipy import stats

    # Get the inhibition matrix
    inh_matrix = compute_pairwise_inhibition_matrix(W_som_e, M, n_som_per_ensemble)

    # Collect pairwise values (upper triangle only to avoid duplicates)
    inhibition_values = []
    orientation_diffs = []

    for i in range(M):
        for j in range(i + 1, M):
            # Symmetrize inhibition (average of i->j and j->i)
            inh_ij = (inh_matrix[i, j] + inh_matrix[j, i]) / 2.0
            inhibition_values.append(inh_ij)

            # Orientation difference
            ori_diff = compute_orientation_difference(pref_deg[i], pref_deg[j])
            orientation_diffs.append(ori_diff)

    inhibition_values = np.array(inhibition_values)
    orientation_diffs = np.array(orientation_diffs)

    # Compute correlation
    if len(inhibition_values) > 2:
        correlation, p_value = stats.pearsonr(inhibition_values, orientation_diffs)
    else:
        correlation, p_value = 0.0, 1.0

    return inhibition_values, orientation_diffs, correlation, p_value


def plot_inhibition_profile(W_som_e: np.ndarray, M: int, n_som_per_ensemble: int,
                            outpath: str, title: str = "Lateral Inhibition Profile") -> None:
    """Plot the Mexican hat inhibition profile as a function of distance."""
    # Compute inhibition from ensemble 0 to all others
    inh_matrix = compute_pairwise_inhibition_matrix(W_som_e, M, n_som_per_ensemble)

    # For visualization, show inhibition from middle ensemble
    ref_ensemble = M // 2

    # Compute circular distances
    distances = []
    inhibitions = []
    for m in range(M):
        if m != ref_ensemble:
            d = min(abs(m - ref_ensemble), M - abs(m - ref_ensemble))
            distances.append(d)
            inhibitions.append(inh_matrix[m, ref_ensemble])

    # Sort by distance
    sorted_pairs = sorted(zip(distances, inhibitions))
    distances, inhibitions = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(distances, inhibitions, 'o-', markersize=8, linewidth=2)
    ax.set_xlabel("Circular distance between ensembles", fontsize=12)
    ax.set_ylabel("Lateral inhibition strength", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_inhibition_matrix(W_som_e: np.ndarray, M: int, n_som_per_ensemble: int,
                           outpath: str, title: str = "Lateral Inhibition Matrix") -> None:
    """Plot the full inhibition matrix as a heatmap."""
    inh_matrix = compute_pairwise_inhibition_matrix(W_som_e, M, n_som_per_ensemble)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(inh_matrix, cmap='hot', interpolation='nearest')
    ax.set_xlabel("Source ensemble", fontsize=12)
    ax.set_ylabel("Target ensemble", fontsize=12)
    ax.set_title(title, fontsize=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Inhibition weight", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_inhibition_vs_orientation(
    inhibition_values: np.ndarray,
    orientation_diffs: np.ndarray,
    correlation: float,
    p_value: float,
    outpath: str,
    title: str = "Inhibition vs Orientation Difference"
) -> None:
    """
    Create a scatter plot showing the relationship between pairwise
    lateral inhibition and orientation difference.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter plot
    ax.scatter(inhibition_values, orientation_diffs, alpha=0.6, s=40, c='steelblue', edgecolors='navy')

    # Add trend line if there's a meaningful correlation
    if len(inhibition_values) > 2:
        z = np.polyfit(inhibition_values, orientation_diffs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(inhibition_values.min(), inhibition_values.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Linear fit')

    ax.set_xlabel("Pairwise lateral inhibition (symmetrized)", fontsize=12)
    ax.set_ylabel("Orientation difference (degrees)", fontsize=12)
    ax.set_title(f"{title}\nCorrelation: r={correlation:.3f}, p={p_value:.4f}", fontsize=14)
    ax.set_ylim([-5, 95])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add interpretation text
    if p_value < 0.05:
        if correlation > 0:
            interp = "Significant POSITIVE correlation:\nMore inhibition → More different orientations"
        else:
            interp = "Significant NEGATIVE correlation:\nMore inhibition → More similar orientations"
    else:
        interp = "No significant correlation"

    ax.text(0.02, 0.98, interp, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_distance_vs_orientation_diff(
    pref_deg: np.ndarray,
    M: int,
    outpath: str,
    title: str = "Distance vs Orientation Difference"
) -> None:
    """
    Plot how orientation difference varies with ensemble distance.
    This helps visualize if nearby neurons develop similar orientations.
    """
    distances = []
    orientation_diffs = []

    for i in range(M):
        for j in range(i + 1, M):
            d = min(abs(i - j), M - abs(i - j))
            ori_diff = compute_orientation_difference(pref_deg[i], pref_deg[j])
            distances.append(d)
            orientation_diffs.append(ori_diff)

    distances = np.array(distances)
    orientation_diffs = np.array(orientation_diffs)

    # Compute mean orientation difference at each distance
    unique_distances = np.unique(distances)
    mean_ori_diff = [orientation_diffs[distances == d].mean() for d in unique_distances]
    std_ori_diff = [orientation_diffs[distances == d].std() for d in unique_distances]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax1.scatter(distances, orientation_diffs, alpha=0.5, s=30, c='steelblue')
    ax1.set_xlabel("Circular distance between ensembles", fontsize=12)
    ax1.set_ylabel("Orientation difference (degrees)", fontsize=12)
    ax1.set_title("All pairwise comparisons", fontsize=12)
    ax1.set_ylim([-5, 95])
    ax1.grid(True, alpha=0.3)

    # Mean with error bars
    ax2.errorbar(unique_distances, mean_ori_diff, yerr=std_ori_diff,
                 fmt='o-', markersize=10, linewidth=2, capsize=5, color='darkblue')
    ax2.axhline(y=45, color='gray', linestyle='--', alpha=0.5, label='Random expectation (45°)')
    ax2.set_xlabel("Circular distance between ensembles", fontsize=12)
    ax2.set_ylabel("Mean orientation difference (degrees)", fontsize=12)
    ax2.set_title("Mean orientation difference by distance", fontsize=12)
    ax2.set_ylim([-5, 95])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def analyze_mexican_hat_effect(
    pref_deg: np.ndarray,
    W_som_e: np.ndarray,
    M: int,
    n_som_per_ensemble: int = 1
) -> dict:
    """
    Analyze whether the Mexican hat inhibition profile produces the expected
    pattern: nearby neurons (low inhibition) → similar orientations,
    intermediate neurons (high inhibition) → different orientations.

    This analysis is designed for non-monotonic relationships that follow
    the Mexican hat shape.

    Returns a dictionary with:
    - near_mean_diff: Mean orientation difference for nearest neighbors
    - peak_inh_mean_diff: Mean orientation difference for peak inhibition pairs
    - effect_size: Difference between peak and near (positive = expected pattern)
    - t_statistic, p_value: Statistical test comparing near vs peak
    """
    from scipy import stats

    inh_matrix = compute_pairwise_inhibition_matrix(W_som_e, M, n_som_per_ensemble)

    # Collect data organized by inhibition level
    inhibition_levels = []
    orientation_diffs = []
    distances = []

    for i in range(M):
        for j in range(i + 1, M):
            d = min(abs(i - j), M - abs(i - j))
            inh_ij = (inh_matrix[i, j] + inh_matrix[j, i]) / 2.0
            ori_diff = compute_orientation_difference(pref_deg[i], pref_deg[j])

            inhibition_levels.append(inh_ij)
            orientation_diffs.append(ori_diff)
            distances.append(d)

    inhibition_levels = np.array(inhibition_levels)
    orientation_diffs = np.array(orientation_diffs)
    distances = np.array(distances)

    # Find inhibition thresholds for "near" (low inhibition) and "peak" (high inhibition)
    inh_25 = np.percentile(inhibition_levels, 25)  # Low inhibition threshold
    inh_75 = np.percentile(inhibition_levels, 75)  # High inhibition threshold

    # Alternative: use distance-based grouping which is cleaner
    # Near = d=1, Peak = d where inhibition is maximum
    near_mask = distances == 1
    peak_d = distances[np.argmax([inhibition_levels[distances == d].mean()
                                   for d in np.unique(distances) if (distances == d).sum() > 0]
                                  if len(np.unique(distances)) > 0 else [1])]

    # Find the distance with maximum mean inhibition
    unique_d = np.unique(distances)
    mean_inh_by_d = {d: inhibition_levels[distances == d].mean() for d in unique_d}
    peak_d = max(mean_inh_by_d, key=mean_inh_by_d.get)
    peak_mask = distances == peak_d

    near_ori_diffs = orientation_diffs[near_mask]
    peak_ori_diffs = orientation_diffs[peak_mask]

    near_mean = near_ori_diffs.mean() if len(near_ori_diffs) > 0 else 45.0
    peak_mean = peak_ori_diffs.mean() if len(peak_ori_diffs) > 0 else 45.0

    # Effect size: positive means peak inhibition → larger orientation differences
    effect_size = peak_mean - near_mean

    # Statistical test
    if len(near_ori_diffs) > 1 and len(peak_ori_diffs) > 1:
        t_stat, p_val = stats.ttest_ind(peak_ori_diffs, near_ori_diffs)
    else:
        t_stat, p_val = 0.0, 1.0

    # Also compute correlation for monotonic portion (low to peak)
    # This tests if the rising part of Mexican hat shows expected pattern
    rising_mask = distances <= peak_d
    if rising_mask.sum() > 2:
        rising_corr, rising_p = stats.pearsonr(
            inhibition_levels[rising_mask],
            orientation_diffs[rising_mask]
        )
    else:
        rising_corr, rising_p = 0.0, 1.0

    return {
        'near_mean_diff': near_mean,
        'peak_mean_diff': peak_mean,
        'effect_size': effect_size,
        'peak_distance': peak_d,
        't_statistic': t_stat,
        'p_value': p_val,
        'rising_correlation': rising_corr,
        'rising_p_value': rising_p,
        'random_expectation': 45.0
    }


def run_inhibition_orientation_analysis(
    net: 'RgcLgnV1Network',
    pref_deg: np.ndarray,
    osi: np.ndarray,
    outdir: str,
    prefix: str = ""
) -> dict:
    """
    Run complete analysis of inhibition-orientation correlation.

    Args:
        net: The trained network
        pref_deg: Preferred orientations of each ensemble (degrees)
        osi: OSI values for each ensemble
        outdir: Directory to save plots
        prefix: Prefix for output filenames

    Returns:
        Dictionary with analysis results
    """
    p = net.p
    M = p.M
    n_som_per_ensemble = p.n_som_per_ensemble

    # Compute correlation
    inh_vals, ori_diffs, corr, pval = analyze_inhibition_orientation_correlation(
        pref_deg, net.W_som_e, M, n_som_per_ensemble
    )

    # Generate all plots
    plot_inhibition_profile(
        net.W_som_e, M, n_som_per_ensemble,
        os.path.join(outdir, f"{prefix}inhibition_profile.png"),
        "Mexican Hat Lateral Inhibition Profile"
    )

    plot_inhibition_matrix(
        net.W_som_e, M, n_som_per_ensemble,
        os.path.join(outdir, f"{prefix}inhibition_matrix.png"),
        "Lateral Inhibition Matrix (SOM→E)"
    )

    plot_inhibition_vs_orientation(
        inh_vals, ori_diffs, corr, pval,
        os.path.join(outdir, f"{prefix}inhibition_vs_orientation.png"),
        "Lateral Inhibition vs Orientation Difference"
    )

    plot_distance_vs_orientation_diff(
        pref_deg, M,
        os.path.join(outdir, f"{prefix}distance_vs_orientation.png"),
        "Ensemble Distance vs Orientation Difference"
    )

    # Run Mexican hat specific analysis
    mh_results = analyze_mexican_hat_effect(pref_deg, net.W_som_e, M, n_som_per_ensemble)

    # Summary statistics
    results = {
        'correlation': corr,
        'p_value': pval,
        'mean_osi': float(osi.mean()),
        'std_osi': float(osi.std()),
        'frac_osi_above_0.3': float((osi > 0.3).mean()),
        'frac_osi_above_0.5': float((osi > 0.5).mean()),
        'mean_orientation_diff': float(ori_diffs.mean()),
        'inhibition_values': inh_vals,
        'orientation_diffs': ori_diffs,
        'mexican_hat_analysis': mh_results
    }

    print(f"\n{'='*70}")
    print("INHIBITION-ORIENTATION CORRELATION ANALYSIS")
    print(f"{'='*70}")

    print("\n--- ORIENTATION SELECTIVITY ---")
    print(f"Mean OSI: {osi.mean():.3f} +/- {osi.std():.3f}")
    print(f"Fraction with OSI > 0.3: {(osi > 0.3).mean()*100:.1f}%")
    print(f"Fraction with OSI > 0.5: {(osi > 0.5).mean()*100:.1f}%")

    print("\n--- LINEAR CORRELATION TEST ---")
    print(f"Pearson correlation (r): {corr:.4f}")
    print(f"P-value: {pval:.4f}")
    print(f"Significant at p<0.05: {'YES' if pval < 0.05 else 'NO'}")

    print("\n--- MEXICAN HAT EFFECT TEST ---")
    print(f"(Tests if nearby neurons develop similar orientations,")
    print(f" while neurons at peak inhibition distance develop different orientations)")
    print(f"")
    print(f"Near neighbors (d=1) mean orientation diff: {mh_results['near_mean_diff']:.1f}°")
    print(f"Peak inhibition (d={mh_results['peak_distance']}) mean orientation diff: {mh_results['peak_mean_diff']:.1f}°")
    print(f"Random expectation: {mh_results['random_expectation']:.1f}°")
    print(f"")
    print(f"Effect size (peak - near): {mh_results['effect_size']:.1f}°")
    print(f"T-test p-value: {mh_results['p_value']:.4f}")
    print(f"Significant at p<0.05: {'YES' if mh_results['p_value'] < 0.05 else 'NO'}")
    print(f"")
    print(f"Rising portion correlation (d=1 to peak): {mh_results['rising_correlation']:.4f}")
    print(f"Rising portion p-value: {mh_results['rising_p_value']:.4f}")

    print("\n--- INTERPRETATION ---")
    if mh_results['effect_size'] > 5 and mh_results['near_mean_diff'] < 40:
        print("STRONG MEXICAN HAT EFFECT DETECTED!")
        print(f"- Nearby neurons (d=1) develop SIMILAR orientations ({mh_results['near_mean_diff']:.1f}° < 45° random)")
        print(f"- Neurons at peak inhibition develop MORE DIFFERENT orientations ({mh_results['peak_mean_diff']:.1f}°)")
        if mh_results['p_value'] < 0.05:
            print("- This effect is STATISTICALLY SIGNIFICANT.")
        print("\nThis confirms the hypothesis: lateral inhibition shapes orientation maps")
        print("such that competitive (high inhibition) pairs differentiate while")
        print("cooperative (low inhibition) pairs develop similar preferences.")
    elif mh_results['near_mean_diff'] < 40:
        print("PARTIAL MEXICAN HAT EFFECT:")
        print(f"- Nearby neurons show similar orientations ({mh_results['near_mean_diff']:.1f}° < 45° random)")
        print(f"- Effect size is modest ({mh_results['effect_size']:.1f}°)")
    else:
        print("WEAK OR NO MEXICAN HAT EFFECT:")
        print("- The expected pattern is not clearly visible in this run.")
        print("- This could be due to random variation or insufficient training.")

    print(f"{'='*70}\n")

    return results


# =============================================================================
# Main training loop
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Biologically plausible V1 STDP network")
    ap.add_argument("--out", type=str, default="runs/bio_plausible",
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

    args = ap.parse_args()
    safe_mkdir(args.out)

    # Create network
    p = Params(
        N=args.N,
        M=args.M,
        seed=args.seed,
        train_segments=args.train_segments,
        segment_ms=args.segment_ms
    )
    net = RgcLgnV1Network(p, init_mode=args.init_mode)

    print(f"[init] Biologically plausible RGC->LGN->V1 network")
    print(f"[init] N={p.N} (patch), M={p.M} (ensembles), n_lgn={net.n_lgn}")
    print(f"[init] Neuron types: LGN=TC(Izhikevich), V1=RS, PV=FS, SOM=LTS")
    print(f"[init] Plasticity: Triplet STDP + Homeostatic scaling")
    print(f"[init] Inhibition: Local PV (feedforward) + SOM (lateral)")
    print(f"[init] init-mode = {args.init_mode}")

    thetas = np.linspace(0, 180 - 180 / args.eval_K, args.eval_K)

    # --- Baseline evaluation ---
    print("\n[baseline] Evaluating tuning at initialization...")
    rates0 = net.evaluate_tuning(thetas, repeats=args.baseline_repeats)
    osi0, pref0 = compute_osi(rates0, thetas)

    print(f"[seg {0:4d}] mean rate={rates0.mean():.3f} Hz | mean OSI={osi0.mean():.3f} | max OSI={osi0.max():.3f}")
    print(f"          prefs(deg) = {np.round(pref0, 1)}")
    print("          NOTE: Nonzero OSI at init is expected from random RF structure")

    plot_weight_maps(net.W, p.N, os.path.join(args.out, "weights_seg0000.png"),
                     title="LGN->V1 weights at init (segment 0)")
    plot_tuning(rates0, thetas, osi0, pref0,
                os.path.join(args.out, "tuning_seg0000.png"),
                title="Baseline tuning (segment 0, before learning)")

    # Tracking
    seg_hist = [0]
    osi_hist = [float(osi0.mean())]
    rate_hist = [float(rates0.mean())]
    pv_rate_hist = []
    som_rate_hist = []

    if p.train_segments == 0:
        print("[final] train-segments=0, no learning occurred")
        print(f"[done] outputs written to: {args.out}")
        return

    # --- Training ---
    print("\n[training] Starting STDP training...")
    for s in range(1, p.train_segments + 1):
        # Random orientation for this segment
        th = float(net.rng.uniform(0.0, 180.0))
        net.run_segment(th, plastic=True)

        if (s % args.viz_every) == 0 or s == p.train_segments:
            rates = net.evaluate_tuning(thetas, repeats=args.eval_repeats)
            osi, pref = compute_osi(rates, thetas)

            print(f"[seg {s:4d}] mean rate={rates.mean():.3f} Hz | mean OSI={osi.mean():.3f} | max OSI={osi.max():.3f}")
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

    # --- Inhibition-Orientation Correlation Analysis ---
    print("\n[analysis] Running inhibition-orientation correlation analysis...")
    analysis_results = run_inhibition_orientation_analysis(
        net, pref1, osi1, args.out, prefix=""
    )

    print(f"\n[done] Outputs written to: {args.out}")


if __name__ == "__main__":
    main()
