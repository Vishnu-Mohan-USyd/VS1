#!/usr/bin/env python3
"""test_inhibition_orientation_correlation.py

Test to analyze whether there is a correlation between lateral inhibition
strength and emergent orientation similarity between V1 neurons.

Hypothesis: Neurons with lesser lateral inhibition (or more lateral excitation)
between them should develop more similar orientation preferences, while neurons
with stronger inhibition should develop more dissimilar (orthogonal) preferences.

This test:
1. Trains the biologically plausible V1 network
2. Computes preferred orientations for all ensembles
3. Calculates pairwise orientation similarity
4. Calculates pairwise lateral connectivity strength
5. Computes and visualizes the correlation

License: MIT
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi, safe_mkdir
)


def orientation_difference(pref1_deg: np.ndarray, pref2_deg: np.ndarray) -> np.ndarray:
    """
    Compute orientation difference accounting for 180-degree periodicity.

    Orientations are in [0, 180) degrees. The circular difference
    gives a value in [0, 90] degrees (max dissimilarity is 90 degrees
    for orthogonal orientations).

    Args:
        pref1_deg: Array of preferred orientations (degrees)
        pref2_deg: Array of preferred orientations (degrees)

    Returns:
        Array of orientation differences in [0, 90] degrees
    """
    diff = np.abs(pref1_deg - pref2_deg)
    # Account for 180-degree periodicity
    diff = np.minimum(diff, 180 - diff)
    return diff


def compute_pairwise_orientation_difference(pref_deg: np.ndarray) -> np.ndarray:
    """
    Compute pairwise orientation difference matrix.

    Args:
        pref_deg: (M,) array of preferred orientations in degrees

    Returns:
        (M, M) matrix where [i,j] is the orientation difference between
        ensembles i and j
    """
    M = len(pref_deg)
    diff_matrix = np.zeros((M, M), dtype=np.float32)

    for i in range(M):
        for j in range(M):
            diff_matrix[i, j] = orientation_difference(
                np.array([pref_deg[i]]),
                np.array([pref_deg[j]])
            )[0]

    return diff_matrix


def compute_net_lateral_connectivity(net: RgcLgnV1Network) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute effective lateral connectivity matrices.

    Returns three matrices:
    1. Lateral excitation (W_e_e): Gaussian distance-dependent
    2. Effective lateral inhibition: Sum of SOM-mediated inhibition paths
    3. Net lateral influence: Excitation - Inhibition

    Args:
        net: The trained network

    Returns:
        Tuple of (excitation, inhibition, net_influence) matrices, each (M, M)
    """
    M = net.M

    # Direct lateral excitation (already computed in network)
    W_exc = net.W_e_e.copy()  # (M, M)

    # SOM-mediated inhibition: E_i -> SOM_i -> E_j (for j != i)
    # W_e_som[som_idx, e_idx]: how much E excites SOM
    # W_som_e[e_idx, som_idx]: how much SOM inhibits E
    # Effective inhibition from E_i to E_j = sum over SOM neurons:
    #   W_e_som[som, i] * W_som_e[j, som]

    # For simplicity with 1 SOM per ensemble:
    # Inhibition from ensemble i to ensemble j = W_e_som[i, i] * W_som_e[j, i]
    W_inh = np.zeros((M, M), dtype=np.float32)

    for i in range(M):
        # SOM neuron index for ensemble i
        som_i = i * net.p.n_som_per_ensemble

        # Drive from E_i to SOM_i
        e_to_som = net.W_e_som[som_i, i]

        for j in range(M):
            if i != j:
                # Inhibition from SOM_i to E_j
                som_to_e = net.W_som_e[j, som_i]
                W_inh[i, j] = e_to_som * som_to_e

    # Net influence (positive = more excitation, negative = more inhibition)
    W_net = W_exc - W_inh

    return W_exc, W_inh, W_net


def extract_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Extract upper triangle elements (excluding diagonal) as 1D array."""
    M = matrix.shape[0]
    indices = np.triu_indices(M, k=1)
    return matrix[indices]


def run_correlation_analysis(
    net: RgcLgnV1Network,
    thetas_deg: np.ndarray,
    eval_repeats: int,
    out_dir: str
) -> dict:
    """
    Run the full correlation analysis.

    Args:
        net: The trained network
        thetas_deg: Array of orientations to test
        eval_repeats: Number of repeats for tuning evaluation
        out_dir: Output directory for plots

    Returns:
        Dictionary with analysis results
    """
    print("\n" + "="*70)
    print("INHIBITION-ORIENTATION CORRELATION ANALYSIS")
    print("="*70)

    # --- Step 1: Evaluate orientation tuning ---
    print("\n[1] Evaluating orientation tuning...")
    rates = net.evaluate_tuning(thetas_deg, repeats=eval_repeats)
    osi, pref_deg = compute_osi(rates, thetas_deg)

    print(f"    OSI range: [{osi.min():.3f}, {osi.max():.3f}], mean: {osi.mean():.3f}")
    print(f"    Preferred orientations: {np.round(pref_deg, 1)}")

    # --- Step 2: Compute pairwise orientation differences ---
    print("\n[2] Computing pairwise orientation differences...")
    ori_diff_matrix = compute_pairwise_orientation_difference(pref_deg)
    ori_diff_pairs = extract_upper_triangle(ori_diff_matrix)

    print(f"    Orientation diff range: [{ori_diff_pairs.min():.1f}, {ori_diff_pairs.max():.1f}] deg")
    print(f"    Mean orientation diff: {ori_diff_pairs.mean():.1f} deg")

    # --- Step 3: Compute lateral connectivity ---
    print("\n[3] Computing lateral connectivity matrices...")
    W_exc, W_inh, W_net = compute_net_lateral_connectivity(net)

    exc_pairs = extract_upper_triangle(W_exc)
    inh_pairs = extract_upper_triangle(W_inh)
    net_pairs = extract_upper_triangle(W_net)

    print(f"    Lateral excitation range: [{exc_pairs.min():.4f}, {exc_pairs.max():.4f}]")
    print(f"    Lateral inhibition range: [{inh_pairs.min():.4f}, {inh_pairs.max():.4f}]")
    print(f"    Net influence range: [{net_pairs.min():.4f}, {net_pairs.max():.4f}]")

    # --- Step 4: Compute correlations ---
    print("\n[4] Computing correlations...")

    # Correlation between excitation and orientation difference
    # Hypothesis: More excitation -> smaller orientation difference
    r_exc, p_exc = stats.pearsonr(exc_pairs, ori_diff_pairs)
    rho_exc, p_rho_exc = stats.spearmanr(exc_pairs, ori_diff_pairs)

    # Correlation between inhibition and orientation difference
    # Hypothesis: More inhibition -> larger orientation difference
    # Note: In this model, SOM-mediated inhibition may be uniform (constant)
    if np.std(inh_pairs) > 1e-9:
        r_inh, p_inh = stats.pearsonr(inh_pairs, ori_diff_pairs)
        rho_inh, p_rho_inh = stats.spearmanr(inh_pairs, ori_diff_pairs)
    else:
        # Inhibition is uniform - no meaningful correlation
        r_inh, p_inh = np.nan, np.nan
        rho_inh, p_rho_inh = np.nan, np.nan
        print("    NOTE: Lateral inhibition is uniform (constant) in this model.")

    # Correlation between net influence and orientation difference
    # Hypothesis: More positive (excitatory) net influence -> smaller orientation difference
    r_net, p_net = stats.pearsonr(net_pairs, ori_diff_pairs)
    rho_net, p_rho_net = stats.spearmanr(net_pairs, ori_diff_pairs)

    print(f"\n    Excitation vs Orientation Difference:")
    print(f"        Pearson r = {r_exc:+.4f} (p = {p_exc:.2e})")
    print(f"        Spearman rho = {rho_exc:+.4f} (p = {p_rho_exc:.2e})")
    print(f"        Interpretation: {'NEGATIVE correlation expected' if r_exc < 0 else 'Positive correlation (unexpected)'}")

    print(f"\n    Inhibition vs Orientation Difference:")
    print(f"        Pearson r = {r_inh:+.4f} (p = {p_inh:.2e})")
    print(f"        Spearman rho = {rho_inh:+.4f} (p = {p_rho_inh:.2e})")
    print(f"        Interpretation: {'POSITIVE correlation expected' if r_inh > 0 else 'Negative correlation (unexpected)'}")

    print(f"\n    Net Influence vs Orientation Difference:")
    print(f"        Pearson r = {r_net:+.4f} (p = {p_net:.2e})")
    print(f"        Spearman rho = {rho_net:+.4f} (p = {p_rho_net:.2e})")
    print(f"        Interpretation: {'NEGATIVE correlation expected' if r_net < 0 else 'Positive correlation (unexpected)'}")

    # --- Step 5: Ensemble distance analysis ---
    print("\n[5] Analyzing ensemble distance vs orientation similarity...")

    # Compute pairwise ensemble distances (circular topology)
    M = net.M
    ensemble_dist = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            d = min(abs(i - j), M - abs(i - j))
            ensemble_dist[i, j] = d

    dist_pairs = extract_upper_triangle(ensemble_dist)

    r_dist, p_dist = stats.pearsonr(dist_pairs, ori_diff_pairs)
    rho_dist, p_rho_dist = stats.spearmanr(dist_pairs, ori_diff_pairs)

    print(f"    Ensemble Distance vs Orientation Difference:")
    print(f"        Pearson r = {r_dist:+.4f} (p = {p_dist:.2e})")
    print(f"        Spearman rho = {rho_dist:+.4f} (p = {p_rho_dist:.2e})")
    print(f"        Interpretation: If positive, nearby ensembles have similar orientations")

    # --- Step 6: Generate visualizations ---
    print("\n[6] Generating visualizations...")

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 14))

    # Subplot 1: Scatter plot - Excitation vs Orientation Difference
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.scatter(exc_pairs, ori_diff_pairs, alpha=0.5, s=20)
    ax1.set_xlabel("Lateral Excitation Strength")
    ax1.set_ylabel("Orientation Difference (deg)")
    ax1.set_title(f"Excitation vs Orientation Diff\nr={r_exc:.3f}, p={p_exc:.2e}")

    # Add regression line
    z = np.polyfit(exc_pairs, ori_diff_pairs, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(exc_pairs.min(), exc_pairs.max(), 100)
    ax1.plot(x_line, p_line(x_line), 'r-', linewidth=2, label='Linear fit')
    ax1.legend()

    # Subplot 2: Scatter plot - Inhibition vs Orientation Difference
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.scatter(inh_pairs, ori_diff_pairs, alpha=0.5, s=20, color='orange')
    ax2.set_xlabel("Lateral Inhibition Strength")
    ax2.set_ylabel("Orientation Difference (deg)")
    ax2.set_title(f"Inhibition vs Orientation Diff\nr={r_inh:.3f}, p={p_inh:.2e}")

    # Subplot 3: Scatter plot - Net Influence vs Orientation Difference
    ax3 = fig.add_subplot(3, 3, 3)
    colors = ['blue' if n > 0 else 'red' for n in net_pairs]
    ax3.scatter(net_pairs, ori_diff_pairs, alpha=0.5, s=20, c=colors)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Net Lateral Influence (Exc - Inh)")
    ax3.set_ylabel("Orientation Difference (deg)")
    ax3.set_title(f"Net Influence vs Orientation Diff\nr={r_net:.3f}, p={p_net:.2e}")

    # Add regression line
    z = np.polyfit(net_pairs, ori_diff_pairs, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(net_pairs.min(), net_pairs.max(), 100)
    ax3.plot(x_line, p_line(x_line), 'g-', linewidth=2, label='Linear fit')
    ax3.legend()

    # Subplot 4: Ensemble Distance vs Orientation Difference
    ax4 = fig.add_subplot(3, 3, 4)
    # Add jitter for better visualization
    jitter = np.random.normal(0, 0.1, len(dist_pairs))
    ax4.scatter(dist_pairs + jitter, ori_diff_pairs, alpha=0.5, s=20, color='green')
    ax4.set_xlabel("Ensemble Distance (circular topology)")
    ax4.set_ylabel("Orientation Difference (deg)")
    ax4.set_title(f"Ensemble Distance vs Orientation Diff\nr={r_dist:.3f}, p={p_dist:.2e}")

    # Box plot for each distance
    unique_dists = np.unique(dist_pairs)
    box_data = [ori_diff_pairs[dist_pairs == d] for d in unique_dists]
    positions = unique_dists
    bp = ax4.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.3)

    # Subplot 5: Orientation Difference Matrix (heatmap)
    ax5 = fig.add_subplot(3, 3, 5)
    im5 = ax5.imshow(ori_diff_matrix, cmap='viridis', aspect='equal')
    ax5.set_xlabel("Ensemble j")
    ax5.set_ylabel("Ensemble i")
    ax5.set_title("Pairwise Orientation Difference (deg)")
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # Subplot 6: Net Lateral Influence Matrix (heatmap)
    ax6 = fig.add_subplot(3, 3, 6)
    im6 = ax6.imshow(W_net, cmap='RdBu_r', aspect='equal',
                     vmin=-np.abs(W_net).max(), vmax=np.abs(W_net).max())
    ax6.set_xlabel("Ensemble j")
    ax6.set_ylabel("Ensemble i")
    ax6.set_title("Net Lateral Influence (Exc - Inh)")
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    # Subplot 7: Circular orientation map
    ax7 = fig.add_subplot(3, 3, 7, projection='polar')
    # Map ensemble index to angle (circular topology)
    ensemble_angles = np.linspace(0, 2*np.pi, M, endpoint=False)
    # Map preferred orientation to color
    colors = plt.cm.hsv(pref_deg / 180.0)
    bars = ax7.bar(ensemble_angles, osi, width=2*np.pi/M * 0.8,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax7.set_title("Orientation Map (color=pref, height=OSI)")
    ax7.set_rticks([0.2, 0.4, 0.6, 0.8])
    ax7.set_rlabel_position(0)

    # Subplot 8: Tuning curves
    ax8 = fig.add_subplot(3, 3, 8)
    for m in range(min(M, 8)):  # Plot first 8 ensembles
        ax8.plot(thetas_deg, rates[m], marker='o', markersize=3,
                label=f"E{m} (pref={pref_deg[m]:.0f})")
    ax8.set_xlabel("Orientation (deg)")
    ax8.set_ylabel("Firing rate (Hz)")
    ax8.set_title("Tuning Curves (first 8 ensembles)")
    ax8.legend(fontsize=6, ncol=2)

    # Subplot 9: Summary statistics
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')

    # Create summary text
    summary_text = f"""
    ANALYSIS SUMMARY
    ================

    Network Parameters:
    - M = {M} ensembles
    - Mean OSI = {osi.mean():.3f}
    - OSI range = [{osi.min():.3f}, {osi.max():.3f}]

    Key Correlations:

    1. Lateral Excitation vs Orientation Diff:
       r = {r_exc:+.4f} (p = {p_exc:.2e})
       {"SIGNIFICANT" if p_exc < 0.05 else "Not significant"}

    2. Lateral Inhibition vs Orientation Diff:
       r = {r_inh:+.4f} (p = {p_inh:.2e})
       {"SIGNIFICANT" if not np.isnan(p_inh) and p_inh < 0.05 else "Not significant"}
       {"(Note: Inhibition is uniform in this model)" if np.isnan(r_inh) else ""}

    3. Net Influence vs Orientation Diff:
       r = {r_net:+.4f} (p = {p_net:.2e})
       {"SIGNIFICANT" if p_net < 0.05 else "Not significant"}

    4. Ensemble Distance vs Orientation Diff:
       r = {r_dist:+.4f} (p = {p_dist:.2e})
       {"SIGNIFICANT" if p_dist < 0.05 else "Not significant"}

    Interpretation:
    {"Nearby ensembles develop SIMILAR orientations (as expected from lateral excitation)" if r_dist > 0 and p_dist < 0.05 else
     "Nearby ensembles develop DIFFERENT orientations (competition dominates)" if r_dist < 0 and p_dist < 0.05 else
     "No significant spatial organization of orientations"}
    """
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "inhibition_orientation_correlation.png"), dpi=150)
    plt.close(fig)
    print(f"    Saved: {os.path.join(out_dir, 'inhibition_orientation_correlation.png')}")

    # --- Additional plot: Binned analysis ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # Bin by distance and compute mean orientation difference
    ax = axes2[0]
    unique_dists = np.unique(dist_pairs)
    mean_ori_diff_by_dist = []
    std_ori_diff_by_dist = []
    for d in unique_dists:
        mask = dist_pairs == d
        mean_ori_diff_by_dist.append(ori_diff_pairs[mask].mean())
        std_ori_diff_by_dist.append(ori_diff_pairs[mask].std())

    ax.errorbar(unique_dists, mean_ori_diff_by_dist, yerr=std_ori_diff_by_dist,
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax.set_xlabel("Ensemble Distance (circular topology)", fontsize=12)
    ax.set_ylabel("Mean Orientation Difference (deg)", fontsize=12)
    ax.set_title("Orientation Similarity vs Ensemble Distance", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Bin by excitation and compute mean orientation difference
    ax = axes2[1]
    n_bins = 5
    exc_bins = np.percentile(exc_pairs[exc_pairs > 0], np.linspace(0, 100, n_bins + 1))
    exc_bin_centers = []
    mean_ori_diff_by_exc = []
    std_ori_diff_by_exc = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (exc_pairs >= exc_bins[i]) & (exc_pairs <= exc_bins[i+1])
        else:
            mask = (exc_pairs >= exc_bins[i]) & (exc_pairs < exc_bins[i+1])
        if mask.sum() > 0:
            exc_bin_centers.append((exc_bins[i] + exc_bins[i+1]) / 2)
            mean_ori_diff_by_exc.append(ori_diff_pairs[mask].mean())
            std_ori_diff_by_exc.append(ori_diff_pairs[mask].std())

    if len(exc_bin_centers) > 0:
        ax.errorbar(exc_bin_centers, mean_ori_diff_by_exc, yerr=std_ori_diff_by_exc,
                    fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8, color='blue')
    ax.set_xlabel("Lateral Excitation Strength", fontsize=12)
    ax.set_ylabel("Mean Orientation Difference (deg)", fontsize=12)
    ax.set_title("Orientation Similarity vs Lateral Excitation", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "inhibition_orientation_binned.png"), dpi=150)
    plt.close(fig2)
    print(f"    Saved: {os.path.join(out_dir, 'inhibition_orientation_binned.png')}")

    # --- Return results ---
    results = {
        'osi': osi,
        'pref_deg': pref_deg,
        'ori_diff_matrix': ori_diff_matrix,
        'W_exc': W_exc,
        'W_inh': W_inh,
        'W_net': W_net,
        'correlations': {
            'excitation': {'r': r_exc, 'p': p_exc, 'rho': rho_exc, 'p_rho': p_rho_exc},
            'inhibition': {'r': r_inh, 'p': p_inh, 'rho': rho_inh, 'p_rho': p_rho_inh},
            'net_influence': {'r': r_net, 'p': p_net, 'rho': rho_net, 'p_rho': p_rho_net},
            'ensemble_distance': {'r': r_dist, 'p': p_dist, 'rho': rho_dist, 'p_rho': p_rho_dist}
        }
    }

    return results


def make_distance_dependent_inhibition(net: RgcLgnV1Network, sigma: float = 4.0) -> None:
    """
    Modify the network to have distance-dependent lateral inhibition.

    Instead of uniform SOM->E inhibition, make it inversely related to distance:
    nearby ensembles have LESS mutual inhibition, distant ones have MORE.

    This creates a Mexican-hat like connectivity pattern that should promote
    nearby ensembles developing similar orientations.

    Args:
        net: The network to modify
        sigma: Spread parameter for distance-dependent inhibition
    """
    M = net.M
    p = net.p

    # Recompute W_som_e with distance-dependent weights
    # Higher inhibition for more distant ensembles
    net.W_som_e = np.zeros((M, net.n_som), dtype=np.float32)

    total_inh = p.w_som_e  # Total inhibition budget

    for m in range(M):
        som_start = m * p.n_som_per_ensemble
        som_end = som_start + p.n_som_per_ensemble

        weights = []
        for other in range(M):
            if other != m:
                # Circular distance
                d = min(abs(m - other), M - abs(m - other))
                # Inhibition increases with distance (sigmoid-like)
                # Nearby ensembles: less inhibition; distant: more inhibition
                w = 1.0 / (1.0 + math.exp(-(d - M/4) / sigma))
                weights.append((other, w))

        # Normalize weights to maintain total inhibition budget
        total_w = sum(w for _, w in weights)
        for other, w in weights:
            net.W_som_e[other, som_start:som_end] = total_inh * w / total_w


def main():
    ap = argparse.ArgumentParser(
        description="Test correlation between lateral inhibition and orientation similarity"
    )
    ap.add_argument("--out", type=str, default="runs/inhibition_orientation_test",
                    help="Output directory")
    ap.add_argument("--train-segments", type=int, default=500,
                    help="Number of training segments")
    ap.add_argument("--segment-ms", type=int, default=300,
                    help="Duration of each segment (ms)")
    ap.add_argument("--N", type=int, default=8,
                    help="Patch size NxN")
    ap.add_argument("--M", type=int, default=32,
                    help="Number of V1 ensembles")
    ap.add_argument("--seed", type=int, default=1,
                    help="Random seed")
    ap.add_argument("--eval-K", type=int, default=12,
                    help="Number of orientations to test")
    ap.add_argument("--eval-repeats", type=int, default=5,
                    help="Number of repeats for evaluation")
    ap.add_argument("--skip-training", action="store_true",
                    help="Skip training and just analyze random weights")
    ap.add_argument("--distance-dependent-inhibition", action="store_true",
                    help="Use distance-dependent lateral inhibition instead of uniform")
    ap.add_argument("--inhibition-sigma", type=float, default=4.0,
                    help="Sigma parameter for distance-dependent inhibition")

    args = ap.parse_args()
    safe_mkdir(args.out)

    print("="*70)
    print("INHIBITION-ORIENTATION CORRELATION TEST")
    print("="*70)
    print(f"Output directory: {args.out}")
    print(f"Training segments: {args.train_segments}")
    print(f"Network: N={args.N}, M={args.M}")
    print(f"Seed: {args.seed}")

    # Create network
    p = Params(
        N=args.N,
        M=args.M,
        seed=args.seed,
        train_segments=args.train_segments,
        segment_ms=args.segment_ms
    )
    net = RgcLgnV1Network(p, init_mode="random")

    print(f"\n[init] Network created with {net.M} ensembles, {net.n_lgn} LGN neurons")

    # Apply distance-dependent inhibition if requested
    if args.distance_dependent_inhibition:
        print(f"[init] Applying distance-dependent lateral inhibition (sigma={args.inhibition_sigma})")
        make_distance_dependent_inhibition(net, sigma=args.inhibition_sigma)
    else:
        print("[init] Using uniform lateral inhibition (default)")

    # Evaluation orientations
    thetas = np.linspace(0, 180 - 180 / args.eval_K, args.eval_K)

    # --- Baseline analysis (before training) ---
    print("\n" + "-"*70)
    print("BASELINE ANALYSIS (before training)")
    print("-"*70)
    baseline_results = run_correlation_analysis(
        net, thetas, args.eval_repeats, args.out
    )

    # Rename baseline outputs
    os.rename(
        os.path.join(args.out, "inhibition_orientation_correlation.png"),
        os.path.join(args.out, "baseline_inhibition_orientation_correlation.png")
    )
    os.rename(
        os.path.join(args.out, "inhibition_orientation_binned.png"),
        os.path.join(args.out, "baseline_inhibition_orientation_binned.png")
    )

    if args.skip_training:
        print("\n[skip] Skipping training as requested")
        print(f"\n[done] Results saved to: {args.out}")
        return baseline_results

    # --- Training ---
    print("\n" + "-"*70)
    print("TRAINING")
    print("-"*70)
    print(f"Training for {args.train_segments} segments...")

    for s in range(1, args.train_segments + 1):
        # Random orientation for this segment
        th = float(net.rng.uniform(0.0, 180.0))
        net.run_segment(th, plastic=True)

        if s % 100 == 0:
            print(f"    Completed segment {s}/{args.train_segments}")

    print("    Training complete!")

    # --- Post-training analysis ---
    print("\n" + "-"*70)
    print("POST-TRAINING ANALYSIS")
    print("-"*70)
    final_results = run_correlation_analysis(
        net, thetas, args.eval_repeats, args.out
    )

    # --- Comparison ---
    print("\n" + "="*70)
    print("COMPARISON: BASELINE vs POST-TRAINING")
    print("="*70)

    print(f"\nMean OSI: {baseline_results['osi'].mean():.3f} -> {final_results['osi'].mean():.3f}")

    print(f"\nExcitation vs Orientation Diff correlation:")
    print(f"    Baseline: r = {baseline_results['correlations']['excitation']['r']:+.4f}")
    print(f"    Final:    r = {final_results['correlations']['excitation']['r']:+.4f}")

    print(f"\nEnsemble Distance vs Orientation Diff correlation:")
    print(f"    Baseline: r = {baseline_results['correlations']['ensemble_distance']['r']:+.4f}")
    print(f"    Final:    r = {final_results['correlations']['ensemble_distance']['r']:+.4f}")

    # --- Final comparison plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Baseline excitation vs ori diff
    ax = axes[0, 0]
    exc_pairs_base = extract_upper_triangle(baseline_results['W_exc'])
    ori_diff_base = extract_upper_triangle(baseline_results['ori_diff_matrix'])
    ax.scatter(exc_pairs_base, ori_diff_base, alpha=0.5, s=20)
    ax.set_xlabel("Lateral Excitation")
    ax.set_ylabel("Orientation Difference (deg)")
    r_base = baseline_results['correlations']['excitation']['r']
    ax.set_title(f"BASELINE: Excitation vs Ori Diff (r={r_base:.3f})")

    # Final excitation vs ori diff
    ax = axes[0, 1]
    exc_pairs_final = extract_upper_triangle(final_results['W_exc'])
    ori_diff_final = extract_upper_triangle(final_results['ori_diff_matrix'])
    ax.scatter(exc_pairs_final, ori_diff_final, alpha=0.5, s=20, color='red')
    ax.set_xlabel("Lateral Excitation")
    ax.set_ylabel("Orientation Difference (deg)")
    r_final = final_results['correlations']['excitation']['r']
    ax.set_title(f"POST-TRAINING: Excitation vs Ori Diff (r={r_final:.3f})")

    # Baseline distance vs ori diff
    ax = axes[1, 0]
    M = net.M
    ensemble_dist = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            d = min(abs(i - j), M - abs(i - j))
            ensemble_dist[i, j] = d
    dist_pairs = extract_upper_triangle(ensemble_dist)

    ax.scatter(dist_pairs + np.random.normal(0, 0.1, len(dist_pairs)),
               ori_diff_base, alpha=0.5, s=20, color='green')
    ax.set_xlabel("Ensemble Distance")
    ax.set_ylabel("Orientation Difference (deg)")
    r_dist_base = baseline_results['correlations']['ensemble_distance']['r']
    ax.set_title(f"BASELINE: Distance vs Ori Diff (r={r_dist_base:.3f})")

    # Final distance vs ori diff
    ax = axes[1, 1]
    ax.scatter(dist_pairs + np.random.normal(0, 0.1, len(dist_pairs)),
               ori_diff_final, alpha=0.5, s=20, color='purple')
    ax.set_xlabel("Ensemble Distance")
    ax.set_ylabel("Orientation Difference (deg)")
    r_dist_final = final_results['correlations']['ensemble_distance']['r']
    ax.set_title(f"POST-TRAINING: Distance vs Ori Diff (r={r_dist_final:.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "comparison_baseline_vs_final.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {os.path.join(args.out, 'comparison_baseline_vs_final.png')}")

    print(f"\n[done] All results saved to: {args.out}")

    return final_results


if __name__ == "__main__":
    main()
