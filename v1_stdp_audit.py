#!/usr/bin/env python3
"""v1_stdp_audit.py

RGC → LGN → V1(L4) spiking network with STDP that can *learn* orientation selectivity.

This file exists for one reason: **auditability**.

Always:
  1) evaluates and plots **baseline tuning at segment 0** (before any plasticity),
  2) evaluates tuning during training, and
  3) reports **ΔOSI = OSI_after − OSI_before** (not just the final OSI).

Hard constraint:
  - No orientation-coded initialization (no Gabor templates, no angle-coded delay gradients).
  - LGN→V1 weights are random, and LGN→V1 conduction delays are random.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_osi(rates_hz: np.ndarray, thetas_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Classic doubled-angle OSI.

    OSI = |Σ r(θ) e^{i2θ}| / Σ r(θ)
    pref = 0.5 * arg(Σ r(θ) e^{i2θ}) in [0,180)
    """
    th = np.deg2rad(thetas_deg)
    vec = (rates_hz * np.exp(1j * 2 * th)[None, :]).sum(axis=1)
    denom = rates_hz.sum(axis=1) + 1e-9
    osi = np.abs(vec) / denom
    pref = (0.5 * np.angle(vec)) % np.pi
    return osi, np.rad2deg(pref)


@dataclass
class Params:
    # Geometry
    N: int = 8          # LGN patch size (N x N)
    M: int = 8          # Number of V1 ensembles
    dt_ms: float = 1.0  # Timestep

    # Training
    segment_ms: int = 300
    train_segments: int = 200
    seed: int = 1

    # Drifting gratings
    spatial_freq: float = 0.12
    temporal_freq: float = 8.0
    base_rate: float = 5.0
    gain_rate: float = 120.0

    # RGC->LGN
    w_rgc_lgn: float = 4.0

    # LGN LIF
    tau_m_lgn: float = 20.0
    tau_syn_lgn: float = 10.0
    v_rest_lgn: float = -65.0
    v_reset_lgn: float = -65.0
    v_th_lgn: float = -50.0
    ref_lgn: int = 2

    # V1 LIF
    tau_m_v1: float = 20.0
    v_rest_v1: float = -65.0
    v_reset_v1: float = -65.0
    v_th_v1: float = -50.0
    ref_v1: int = 2

    # Global inhibition
    tau_m_inh: float = 10.0
    tau_syn_inh: float = 10.0
    v_rest_inh: float = -65.0
    v_reset_inh: float = -65.0
    v_th_inh: float = -50.0
    ref_inh: int = 2
    w_ei: float = 4.0
    w_ie: float = 3.0
    tau_inh: float = 8.0

    # Threshold adaptation
    theta_tau: float = 200.0
    theta_inc: float = 0.5

    # LGN->V1 weights & delays
    delay_max: int = 12
    w_init_mean: float = 0.08
    w_init_std: float = 0.03
    w_max: float = 0.40
    w_norm_target: float = 10.0

    # STDP
    tau_pre: float = 20.0
    tau_post: float = 20.0
    A_plus: float = 0.008
    A_minus: float = 0.010
    w_decay: float = 0.0001


class RgcLgnV1:
    def __init__(self, p: Params, *, init_mode: str = "random"):
        self.p = p
        self.rng = np.random.default_rng(p.seed)

        self.N = p.N
        self.pre = 2 * p.N * p.N  # ON + OFF channels
        self.M = p.M
        self.L = p.delay_max + 1

        # Spatial coordinates for grating
        xs = np.arange(p.N) - (p.N - 1) / 2.0
        ys = np.arange(p.N) - (p.N - 1) / 2.0
        self.X, self.Y = np.meshgrid(xs, ys, indexing="xy")

        # LGN state
        self.V_lgn = np.full(self.pre, p.v_rest_lgn, dtype=np.float32)
        self.I_lgn = np.zeros(self.pre, dtype=np.float32)
        self.refcnt_lgn = np.zeros(self.pre, dtype=np.int16)

        # V1 state
        self.V_v1 = np.full(p.M, p.v_rest_v1, dtype=np.float32)
        self.refcnt_v1 = np.zeros(p.M, dtype=np.int16)
        self.theta = np.zeros(p.M, dtype=np.float32)  # Threshold adaptation

        # Inhibitory interneuron state (single global inhibitory neuron)
        self.V_inh = np.float32(p.v_rest_inh)
        self.I_inh_syn = np.float32(0.0)
        self.refcnt_inh = np.int16(0)
        self.I_inh_to_v1 = np.zeros(p.M, dtype=np.float32)

        # Delay buffer for LGN spikes
        self.buf = np.zeros((self.L, self.pre), dtype=np.uint8)
        self.ptr = 0
        self.pre_ids = np.arange(self.pre)[None, :]

        # Unbiased, random delays (integers in [0, delay_max])
        self.D = self.rng.integers(0, self.L, size=(p.M, self.pre), endpoint=False, dtype=np.int16)

        # Unbiased weights
        if init_mode == "random":
            W = self.rng.normal(p.w_init_mean, p.w_init_std, size=(p.M, self.pre)).astype(np.float32)
        elif init_mode == "near_uniform":
            W = (p.w_init_mean + self.rng.normal(0.0, p.w_init_std * 0.05, size=(p.M, self.pre))).astype(np.float32)
        else:
            raise ValueError("init_mode must be 'random' or 'near_uniform'")

        self.W = np.clip(W, 0.0, p.w_max)

        # STDP traces
        self.x_pre = np.zeros((p.M, self.pre), dtype=np.float32)
        self.x_post = np.zeros(p.M, dtype=np.float32)

        # Precompute decay factors
        self.decay_pre = math.exp(-p.dt_ms / p.tau_pre)
        self.decay_post = math.exp(-p.dt_ms / p.tau_post)
        self.decay_lgn_syn = math.exp(-p.dt_ms / p.tau_syn_lgn)
        self.decay_inh_syn = math.exp(-p.dt_ms / p.tau_syn_inh)
        self.decay_inh_to_v1 = math.exp(-p.dt_ms / p.tau_inh)
        self.decay_theta = math.exp(-p.dt_ms / p.theta_tau)

    def reset_state(self) -> None:
        """Reset all dynamic state (but keep weights)."""
        p = self.p
        self.V_lgn.fill(p.v_rest_lgn)
        self.I_lgn.fill(0.0)
        self.refcnt_lgn.fill(0)

        self.V_v1.fill(p.v_rest_v1)
        self.refcnt_v1.fill(0)
        self.theta.fill(0.0)

        self.V_inh = np.float32(p.v_rest_inh)
        self.I_inh_syn = np.float32(0.0)
        self.refcnt_inh = np.int16(0)
        self.I_inh_to_v1.fill(0.0)

        self.buf.fill(0)
        self.ptr = 0

        self.x_pre.fill(0.0)
        self.x_post.fill(0.0)

    def grating(self, theta_deg: float, t_ms: float, phase: float) -> np.ndarray:
        """Generate drifting grating stimulus."""
        p = self.p
        th = math.radians(theta_deg)
        coord = self.X * math.cos(th) + self.Y * math.sin(th)
        return np.sin(2 * math.pi * (p.spatial_freq * coord - p.temporal_freq * (t_ms / 1000.0)) + phase).astype(np.float32)

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
        """Single simulation step."""
        p = self.p
        rgc = np.concatenate([on_spk.ravel(), off_spk.ravel()]).astype(np.uint8)

        # === LGN dynamics ===
        self.I_lgn *= self.decay_lgn_syn
        self.I_lgn += p.w_rgc_lgn * rgc

        self.refcnt_lgn[self.refcnt_lgn > 0] -= 1
        active = (self.refcnt_lgn == 0)
        self.V_lgn[active] += (p.dt_ms / p.tau_m_lgn) * (p.v_rest_lgn - self.V_lgn[active]) + self.I_lgn[active]
        self.V_lgn[~active] = p.v_reset_lgn

        lgn_spk = (self.V_lgn >= p.v_th_lgn) & active
        if lgn_spk.any():
            self.V_lgn[lgn_spk] = p.v_reset_lgn
            self.refcnt_lgn[lgn_spk] = p.ref_lgn

        # Store in delay buffer
        lgn_spk_u8 = lgn_spk.astype(np.uint8)
        self.buf[self.ptr, :] = lgn_spk_u8

        # Get delayed arrivals for each V1 neuron
        idx = (self.ptr - self.D) % self.L
        arrivals = self.buf[idx, self.pre_ids].astype(np.float32)  # (M, pre)

        # === STDP (pre-spike part) ===
        if plastic:
            self.x_pre *= self.decay_pre
            self.x_post *= self.decay_post

            # LTD: pre spike with recent post activity
            if arrivals.any():
                self.W -= p.A_minus * arrivals * self.x_post[:, None] * self.W
            self.x_pre += arrivals

        # === V1 dynamics ===
        I_ff = (self.W * arrivals).sum(axis=1)
        self.I_inh_to_v1 *= self.decay_inh_to_v1

        self.refcnt_v1[self.refcnt_v1 > 0] -= 1
        active_v1 = (self.refcnt_v1 == 0)
        self.theta *= self.decay_theta
        self.V_v1[active_v1] += (p.dt_ms / p.tau_m_v1) * (p.v_rest_v1 - self.V_v1[active_v1]) + I_ff[active_v1] - self.I_inh_to_v1[active_v1]
        self.V_v1[~active_v1] = p.v_reset_v1

        thr_eff = p.v_th_v1 + self.theta
        v1_spk = (self.V_v1 >= thr_eff) & active_v1
        if v1_spk.any():
            self.V_v1[v1_spk] = p.v_reset_v1
            self.refcnt_v1[v1_spk] = p.ref_v1
            self.theta[v1_spk] += p.theta_inc
        v1_spk_u8 = v1_spk.astype(np.uint8)

        # === Global inhibition ===
        self.I_inh_syn *= self.decay_inh_syn
        self.I_inh_syn += p.w_ei * v1_spk_u8.sum()
        if self.refcnt_inh > 0:
            self.refcnt_inh -= 1
            self.V_inh = np.float32(p.v_reset_inh)
            inh_spk = False
        else:
            self.V_inh += (p.dt_ms / p.tau_m_inh) * (p.v_rest_inh - self.V_inh) + self.I_inh_syn
            inh_spk = bool(self.V_inh >= p.v_th_inh)
            if inh_spk:
                self.V_inh = np.float32(p.v_reset_inh)
                self.refcnt_inh = np.int16(p.ref_inh)
        if inh_spk:
            self.I_inh_to_v1 += p.w_ie

        # === STDP (post-spike part) ===
        if plastic:
            # LTP: post spike with recent pre activity (multiplicative)
            if v1_spk.any():
                self.W += p.A_plus * v1_spk_u8[:, None] * self.x_pre * (p.w_max - self.W)
            self.x_post += v1_spk_u8.astype(np.float32)
            # Weight decay
            self.W *= (1.0 - p.w_decay)
            np.clip(self.W, 0.0, p.w_max, out=self.W)

        self.ptr = (self.ptr + 1) % self.L
        return v1_spk_u8

    def run_segment(self, theta_deg: float, plastic: bool) -> np.ndarray:
        """Run one stimulus segment (300ms default)."""
        p = self.p
        steps = int(p.segment_ms / p.dt_ms)
        phase = float(self.rng.uniform(0, 2 * math.pi))
        v1_counts = np.zeros(self.M, dtype=np.int32)

        for k in range(steps):
            stim = self.grating(theta_deg, t_ms=k * p.dt_ms, phase=phase)
            on_spk, off_spk = self.rgc_spikes(stim)
            v1_counts += self.step(on_spk, off_spk, plastic=plastic)

        # Weight normalization after plastic segment
        if plastic:
            s = self.W.sum(axis=1, keepdims=True) + 1e-6
            self.W *= (p.w_norm_target / s)
            np.clip(self.W, 0.0, p.w_max, out=self.W)

        return v1_counts

    def evaluate_tuning(self, thetas_deg: np.ndarray, repeats: int) -> np.ndarray:
        """Evaluate tuning curves without affecting training state."""
        p = self.p
        rates = np.zeros((self.M, len(thetas_deg)), dtype=np.float32)

        # Save state (including RNG)
        rng_state = self.rng.bit_generator.state
        saved = (
            self.V_lgn.copy(), self.I_lgn.copy(), self.refcnt_lgn.copy(),
            self.V_v1.copy(), self.refcnt_v1.copy(), self.theta.copy(),
            float(self.V_inh), float(self.I_inh_syn), int(self.refcnt_inh),
            self.I_inh_to_v1.copy(), self.buf.copy(), int(self.ptr),
            self.x_pre.copy(), self.x_post.copy(),
        )

        for j, th in enumerate(thetas_deg):
            cnt = np.zeros(self.M, dtype=np.float32)
            for _ in range(repeats):
                self.reset_state()
                cnt += self.run_segment(float(th), plastic=False)
            rates[:, j] = cnt / (repeats * (p.segment_ms / 1000.0))

        # Restore state
        (
            self.V_lgn, self.I_lgn, self.refcnt_lgn,
            self.V_v1, self.refcnt_v1, self.theta,
            self.V_inh, self.I_inh_syn, self.refcnt_inh,
            self.I_inh_to_v1, self.buf, self.ptr,
            self.x_pre, self.x_post,
        ) = saved
        self.rng.bit_generator.state = rng_state
        return rates


def plot_weight_maps(W: np.ndarray, N: int, outpath: str, title: str) -> None:
    """Plot ON, OFF, and ON-OFF weight maps."""
    M = W.shape[0]
    W_on = W[:, : N * N].reshape(M, N, N)
    W_off = W[:, N * N :].reshape(M, N, N)
    W_diff = W_on - W_off

    fig, axes = plt.subplots(M, 3, figsize=(9, 2.1 * M))
    if M == 1:
        axes = np.array([axes])

    for m in range(M):
        for j, (arr, coltitle) in enumerate([(W_on[m], "ON"), (W_off[m], "OFF"), (W_diff[m], "ON-OFF")]):
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


def plot_tuning(rates: np.ndarray, thetas_deg: np.ndarray, osi: np.ndarray, pref_deg: np.ndarray, outpath: str, title: str) -> None:
    """Plot tuning curves for all ensembles."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in range(rates.shape[0]):
        ax.plot(thetas_deg, rates[m], marker="o", label=f"E{m} OSI={osi[m]:.2f} pref={pref_deg[m]:.0f}")
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_scalar_over_time(xs: np.ndarray, ys: np.ndarray, outpath: str, ylabel: str, title: str) -> None:
    """Plot a scalar value over training segments."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Training segment")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/v1_audit", help="output directory")
    ap.add_argument("--train-segments", type=int, default=Params.train_segments)
    ap.add_argument("--segment-ms", type=int, default=Params.segment_ms)
    ap.add_argument("--N", type=int, default=Params.N)
    ap.add_argument("--M", type=int, default=Params.M)
    ap.add_argument("--seed", type=int, default=Params.seed)
    ap.add_argument("--viz-every", type=int, default=50)
    ap.add_argument("--eval-K", type=int, default=12)
    ap.add_argument("--eval-repeats", type=int, default=3)
    ap.add_argument("--baseline-repeats", type=int, default=7)
    ap.add_argument("--init-mode", type=str, default="random", choices=["random", "near_uniform"])

    args = ap.parse_args()
    safe_mkdir(args.out)

    p = Params(N=args.N, M=args.M, seed=args.seed, train_segments=args.train_segments, segment_ms=args.segment_ms)
    net = RgcLgnV1(p, init_mode=args.init_mode)

    print(f"[init] N={p.N} (patch), M={p.M} (ensembles), pre={net.pre}, delay_max={p.delay_max} ms")
    print("[init] LGN->V1 weights are UNBIASED (no orientation templates).")
    print("[init] LGN->V1 delays are UNBIASED random integers in [0, delay_max].")
    print(f"[init] init-mode = {args.init_mode}")

    thetas = np.linspace(0, 180 - 180 / args.eval_K, args.eval_K)

    # === Baseline audit at segment 0 ===
    rates0 = net.evaluate_tuning(thetas, repeats=args.baseline_repeats)
    osi0, pref0 = compute_osi(rates0, thetas)

    print(f"[seg {0:4d}] mean rate={rates0.mean():.3f} Hz | mean OSI={osi0.mean():.3f} | max OSI={osi0.max():.3f}")
    print("          prefs(deg) =", np.round(pref0, 1))
    print("          NOTE: Nonzero OSI at init does NOT imply hardcoded orientation; a random RF can be tuned to gratings.")

    plot_weight_maps(net.W, p.N, os.path.join(args.out, "weights_seg0000.png"), title="LGN→V1 weights at init (segment 0)")
    plot_tuning(rates0, thetas, osi0, pref0, os.path.join(args.out, "tuning_seg0000.png"), title="Baseline tuning (segment 0, before learning)")

    seg_hist = [0]
    osi_hist = [float(osi0.mean())]
    rate_hist = [float(rates0.mean())]

    if p.train_segments == 0:
        print("[final] train-segments = 0, so NO learning occurred. The above is baseline tuning only.")
        print(f"[done] outputs written to: {args.out}")
        return

    # === Training ===
    for s in range(1, p.train_segments + 1):
        th = float(net.rng.uniform(0.0, 180.0))
        net.run_segment(th, plastic=True)

        if (s % args.viz_every) == 0 or s == p.train_segments:
            rates = net.evaluate_tuning(thetas, repeats=args.eval_repeats)
            osi, pref = compute_osi(rates, thetas)

            print(f"[seg {s:4d}] mean rate={rates.mean():.3f} Hz | mean OSI={osi.mean():.3f} | max OSI={osi.max():.3f}")
            print("          prefs(deg) =", np.round(pref, 1))

            plot_weight_maps(net.W, p.N, os.path.join(args.out, f"weights_seg{s:04d}.png"), title=f"LGN→V1 weights (segment {s})")
            plot_tuning(rates, thetas, osi, pref, os.path.join(args.out, f"tuning_seg{s:04d}.png"), title=f"Tuning during training (segment {s})")

            seg_hist.append(int(s))
            osi_hist.append(float(osi.mean()))
            rate_hist.append(float(rates.mean()))

    # === Final evaluation ===
    final_repeats = max(args.baseline_repeats, args.eval_repeats, 7)
    rates1 = net.evaluate_tuning(thetas, repeats=final_repeats)
    osi1, pref1 = compute_osi(rates1, thetas)

    d_osi = osi1 - osi0
    print(f"[final] baseline mean OSI={osi0.mean():.3f}  →  final mean OSI={osi1.mean():.3f}  (Δ={d_osi.mean():+.3f})")
    print(f"[final] fraction ensembles with OSI>0.3: {(osi1>0.3).mean()*100:.1f}%")

    plot_weight_maps(net.W, p.N, os.path.join(args.out, "weights_final.png"), title="LGN→V1 weights (final)")
    plot_tuning(rates1, thetas, osi1, pref1, os.path.join(args.out, "tuning_final.png"), title="Final tuning (after learning)")
    plot_scalar_over_time(np.array(seg_hist), np.array(osi_hist), os.path.join(args.out, "mean_osi_over_time.png"), ylabel="mean OSI", title="Mean OSI over training")
    plot_scalar_over_time(np.array(seg_hist), np.array(rate_hist), os.path.join(args.out, "mean_rate_over_time.png"), ylabel="mean rate (Hz)", title="Mean firing rate over training")

    print(f"[done] outputs written to: {args.out}")


if __name__ == "__main__":
    main()
