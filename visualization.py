"""
Visualization and analysis tools for the V1 model.

Includes:
- Orientation Selectivity Index (OSI) computation
- Weight visualization
- Training progress plots
- Tuning curve analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

# Create output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_osi(responses, orientations):
    """
    Compute Orientation Selectivity Index using vector averaging.

    OSI = |sum(R * exp(2i*theta))| / sum(R)

    where R are the responses and theta are orientations.
    The factor of 2 accounts for the 180-degree periodicity of orientation.

    Args:
        responses: Array of responses to each orientation
        orientations: Array of orientations in degrees

    Returns:
        OSI value between 0 (no selectivity) and 1 (perfect selectivity)
    """
    if np.sum(responses) == 0:
        return 0.0

    # Convert to radians and double for 180-degree periodicity
    theta = np.radians(orientations) * 2

    # Complex vector sum
    vector_sum = np.sum(responses * np.exp(1j * theta))

    # OSI is magnitude divided by sum of responses
    osi = np.abs(vector_sum) / np.sum(responses)

    return osi


def compute_preferred_orientation(responses, orientations):
    """
    Compute preferred orientation using vector averaging.

    Args:
        responses: Array of responses to each orientation
        orientations: Array of orientations in degrees

    Returns:
        Preferred orientation in degrees (0-180)
    """
    if np.sum(responses) == 0:
        return 0.0

    # Convert to radians and double
    theta = np.radians(orientations) * 2

    # Complex vector sum
    vector_sum = np.sum(responses * np.exp(1j * theta))

    # Angle is half of the complex angle (due to doubling)
    pref_ori = np.angle(vector_sum) / 2
    pref_ori = np.degrees(pref_ori)

    # Ensure in range [0, 180)
    if pref_ori < 0:
        pref_ori += 180

    return pref_ori


def plot_weight_heatmap(weights, lgn_size, n_hypercolumns, n_orientations,
                        hypercolumn_idx=0, title="LGN-V1 Weights", save_path=None):
    """
    Plot LGN->V1 weights as heatmaps for each orientation in a hypercolumn.

    Args:
        weights: Full weight matrix (n_lgn x n_v1)
        lgn_size: Size of LGN (lgn_size x lgn_size)
        n_hypercolumns: Number of hypercolumns
        n_orientations: Number of orientations per hypercolumn
        hypercolumn_idx: Which hypercolumn to visualize
        title: Plot title
        save_path: Path to save figure (if None, display instead)
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"{title} - Hypercolumn {hypercolumn_idx}", fontsize=14)

    # Reshape weights for this hypercolumn
    n_lgn = lgn_size * lgn_size * 2  # ON + OFF
    neurons_per_ensemble = weights.shape[1] // (n_hypercolumns * n_orientations)

    for ori_idx in range(n_orientations):
        ax = axes.flat[ori_idx]

        # Get V1 neuron indices for this ensemble
        start_idx = (hypercolumn_idx * n_orientations + ori_idx) * neurons_per_ensemble
        end_idx = start_idx + neurons_per_ensemble
        v1_indices = np.arange(start_idx, end_idx)

        # Sum weights to this ensemble
        ensemble_weights = np.sum(weights[:, v1_indices], axis=1)

        # Separate ON and OFF
        on_weights = ensemble_weights[:lgn_size * lgn_size].reshape(lgn_size, lgn_size)
        off_weights = ensemble_weights[lgn_size * lgn_size:].reshape(lgn_size, lgn_size)

        # Combined visualization (ON positive, OFF negative)
        combined = on_weights - off_weights

        im = ax.imshow(combined, cmap='RdBu_r', aspect='equal')
        ax.set_title(f"Ori {ori_idx * 180 // n_orientations}°")
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ensemble_weight_summary(ensemble_weights, n_orientations, title="Ensemble Weights",
                                  save_path=None):
    """
    Plot summary of weights across all ensembles.

    Args:
        ensemble_weights: Array of shape (n_hypercolumns, n_orientations)
        n_orientations: Number of orientations
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Heatmap of all ensemble weights
    ax = axes[0]
    im = ax.imshow(ensemble_weights, aspect='auto', cmap='viridis')
    ax.set_xlabel("Orientation Index")
    ax.set_ylabel("Hypercolumn Index")
    ax.set_title("Weight per Ensemble")
    ax.set_xticks(range(n_orientations))
    ax.set_xticklabels([f"{i * 180 // n_orientations}°" for i in range(n_orientations)])
    plt.colorbar(im, ax=ax)

    # 2. Mean tuning curve across hypercolumns
    ax = axes[1]
    mean_weights = np.mean(ensemble_weights, axis=0)
    std_weights = np.std(ensemble_weights, axis=0)
    orientations = np.arange(n_orientations) * 180 / n_orientations

    ax.bar(orientations, mean_weights, width=15, yerr=std_weights, capsize=3)
    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel("Mean Total Weight")
    ax.set_title("Mean Weight by Orientation")
    ax.set_xticks(orientations)

    # 3. Distribution of weight differences within hypercolumns
    ax = axes[2]
    weight_ranges = np.max(ensemble_weights, axis=1) - np.min(ensemble_weights, axis=1)
    ax.hist(weight_ranges, bins=20, edgecolor='black')
    ax.set_xlabel("Max-Min Weight Difference")
    ax.set_ylabel("Count")
    ax.set_title("Weight Diversity per Hypercolumn")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_osi_distribution(osi_values, title="OSI Distribution", save_path=None):
    """
    Plot distribution of Orientation Selectivity Index values.

    Args:
        osi_values: Array of OSI values
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(osi_values, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(osi_values), color='red', linestyle='--',
               label=f'Mean: {np.mean(osi_values):.3f}')
    ax.axvline(np.median(osi_values), color='green', linestyle='--',
               label=f'Median: {np.median(osi_values):.3f}')
    ax.set_xlabel("OSI")
    ax.set_ylabel("Count")
    ax.set_title("OSI Distribution")
    ax.legend()
    ax.set_xlim(0, 1)

    # Cumulative distribution
    ax = axes[1]
    sorted_osi = np.sort(osi_values)
    cumulative = np.arange(1, len(sorted_osi) + 1) / len(sorted_osi)
    ax.plot(sorted_osi, cumulative)
    ax.set_xlabel("OSI")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("Cumulative OSI Distribution")
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_tuning_curves(responses_by_ensemble, orientations, n_examples=8,
                       title="Tuning Curves", save_path=None):
    """
    Plot example tuning curves from ensembles.

    Args:
        responses_by_ensemble: Array of shape (n_ensembles, n_orientations)
        orientations: Array of orientations in degrees
        n_examples: Number of example curves to plot
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Select random ensembles with reasonable activity
    active_ensembles = np.where(np.max(responses_by_ensemble, axis=1) > 0)[0]
    if len(active_ensembles) == 0:
        print("No active ensembles to plot")
        return

    # Sort by OSI to show range
    osis = [compute_osi(responses_by_ensemble[i], orientations)
            for i in active_ensembles]
    sorted_indices = np.argsort(osis)

    # Select evenly spaced examples
    step = max(1, len(sorted_indices) // n_examples)
    example_indices = [active_ensembles[sorted_indices[i * step]]
                      for i in range(min(n_examples, len(sorted_indices)))]

    for ax_idx, ens_idx in enumerate(example_indices):
        if ax_idx >= 8:
            break

        ax = axes.flat[ax_idx]
        responses = responses_by_ensemble[ens_idx]
        osi = compute_osi(responses, orientations)
        pref_ori = compute_preferred_orientation(responses, orientations)

        ax.plot(orientations, responses, 'o-', linewidth=2, markersize=8)
        ax.axvline(pref_ori, color='red', linestyle='--', alpha=0.5,
                  label=f'Pref: {pref_ori:.0f}°')
        ax.set_xlabel("Orientation (°)")
        ax.set_ylabel("Response")
        ax.set_title(f"Ensemble {ens_idx}\nOSI: {osi:.3f}")
        ax.legend(fontsize=8)
        ax.set_xticks(orientations)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_orientation_map(pref_orientations, osi_values, n_hypercolumns_x, n_hypercolumns_y,
                         title="Orientation Map", save_path=None):
    """
    Plot orientation preference map colored by preferred orientation.

    Args:
        pref_orientations: Array of preferred orientations (n_hypercolumns * n_orientations,)
        osi_values: Array of OSI values
        n_hypercolumns_x: Number of hypercolumns in x
        n_hypercolumns_y: Number of hypercolumns in y
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_orientations = len(pref_orientations) // (n_hypercolumns_x * n_hypercolumns_y)

    # Reshape for visualization
    pref_map = pref_orientations.reshape(n_hypercolumns_x, n_hypercolumns_y, n_orientations)
    osi_map = osi_values.reshape(n_hypercolumns_x, n_hypercolumns_y, n_orientations)

    # Take max OSI orientation for each hypercolumn
    best_ori_idx = np.argmax(osi_map, axis=2)
    best_pref = np.zeros((n_hypercolumns_x, n_hypercolumns_y))
    best_osi = np.zeros((n_hypercolumns_x, n_hypercolumns_y))

    for i in range(n_hypercolumns_x):
        for j in range(n_hypercolumns_y):
            idx = best_ori_idx[i, j]
            best_pref[i, j] = pref_map[i, j, idx]
            best_osi[i, j] = osi_map[i, j, idx]

    # Orientation map
    ax = axes[0]
    im = ax.imshow(best_pref, cmap='hsv', vmin=0, vmax=180, aspect='equal')
    ax.set_title("Preferred Orientation")
    plt.colorbar(im, ax=ax, label="Orientation (°)")

    # OSI map
    ax = axes[1]
    im = ax.imshow(best_osi, cmap='hot', vmin=0, vmax=1, aspect='equal')
    ax.set_title("Orientation Selectivity Index")
    plt.colorbar(im, ax=ax, label="OSI")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_progress(history, save_path=None):
    """
    Plot training progress metrics.

    Args:
        history: Dictionary with training history
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Spike rates over time
    ax = fig.add_subplot(gs[0, 0])
    if 'lgn_rates' in history:
        ax.plot(history['time'], history['lgn_rates'], label='LGN', alpha=0.7)
    if 'v1_exc_rates' in history:
        ax.plot(history['time'], history['v1_exc_rates'], label='V1 Exc', alpha=0.7)
    if 'v1_inh_rates' in history:
        ax.plot(history['time'], history['v1_inh_rates'], label='V1 Inh', alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Spike Rate (Hz)")
    ax.set_title("Population Firing Rates")
    ax.legend()

    # Weight statistics
    ax = fig.add_subplot(gs[0, 1])
    if 'mean_weight' in history:
        ax.plot(history['time'], history['mean_weight'], label='Mean')
    if 'max_weight' in history:
        ax.plot(history['time'], history['max_weight'], label='Max')
    if 'min_weight' in history:
        ax.plot(history['time'], history['min_weight'], label='Min')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Weight")
    ax.set_title("LGN-V1 Weight Statistics")
    ax.legend()

    # OSI over time
    ax = fig.add_subplot(gs[0, 2])
    if 'mean_osi' in history:
        ax.plot(history['time'], history['mean_osi'], label='Mean OSI')
    if 'max_osi' in history:
        ax.plot(history['time'], history['max_osi'], label='Max OSI', alpha=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("OSI")
    ax.set_title("Orientation Selectivity")
    ax.legend()
    ax.set_ylim(0, 1)

    # STDP potentiation/depression
    ax = fig.add_subplot(gs[1, 0])
    if 'total_pot' in history:
        ax.plot(history['time'], history['total_pot'], label='Potentiation')
    if 'total_dep' in history:
        ax.plot(history['time'], history['total_dep'], label='Depression')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Cumulative Change")
    ax.set_title("STDP Activity")
    ax.legend()

    # Weight histogram evolution
    ax = fig.add_subplot(gs[1, 1:])
    if 'weight_hist' in history and len(history['weight_hist']) > 0:
        times = history['weight_hist_times']
        for i, (t, hist, bins) in enumerate(zip(times, history['weight_hist'],
                                                history['weight_bins'])):
            alpha = 0.3 + 0.7 * i / len(times)
            ax.plot(bins[:-1], hist, alpha=alpha, label=f't={t:.0f}ms')
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
        ax.set_title("Weight Distribution Evolution")
        ax.legend(loc='upper right')

    plt.suptitle("Training Progress", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_training_summary(network, responses_by_ensemble, orientations,
                           epoch, save_dir=OUTPUT_DIR):
    """
    Create a comprehensive training summary with multiple plots.

    Args:
        network: The VisualCortexNetwork instance
        responses_by_ensemble: Measured responses (n_ensembles, n_orientations)
        orientations: Array of orientations
        epoch: Current epoch number
        save_dir: Directory to save plots
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{save_dir}/epoch{epoch}_{timestamp}"

    # 1. Weight heatmaps for a few hypercolumns
    weights = network.get_lgn_v1_weights()
    for hc in [0, network.n_hypercolumns // 2, network.n_hypercolumns - 1]:
        if hc < network.n_hypercolumns:
            plot_weight_heatmap(
                weights, network.retina_size, network.n_hypercolumns,
                network.n_orientations, hypercolumn_idx=hc,
                title=f"Epoch {epoch}",
                save_path=f"{prefix}_weights_hc{hc}.png"
            )

    # 2. Ensemble weight summary
    ensemble_weights = network.get_all_ensemble_weights()
    plot_ensemble_weight_summary(
        ensemble_weights, network.n_orientations,
        title=f"Epoch {epoch}",
        save_path=f"{prefix}_ensemble_weights.png"
    )

    # 3. OSI distribution
    osi_values = np.array([compute_osi(responses_by_ensemble[i], orientations)
                          for i in range(len(responses_by_ensemble))])
    plot_osi_distribution(
        osi_values,
        title=f"Epoch {epoch}",
        save_path=f"{prefix}_osi_dist.png"
    )

    # 4. Tuning curves
    plot_tuning_curves(
        responses_by_ensemble, orientations,
        title=f"Epoch {epoch}",
        save_path=f"{prefix}_tuning_curves.png"
    )

    # 5. Orientation map
    pref_oris = np.array([compute_preferred_orientation(responses_by_ensemble[i], orientations)
                         for i in range(len(responses_by_ensemble))])
    plot_orientation_map(
        pref_oris, osi_values,
        network.n_hypercolumns_x, network.n_hypercolumns_y,
        title=f"Epoch {epoch}",
        save_path=f"{prefix}_ori_map.png"
    )

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Summary")
    print(f"{'='*60}")
    print(f"Mean OSI: {np.mean(osi_values):.4f}")
    print(f"Median OSI: {np.median(osi_values):.4f}")
    print(f"Max OSI: {np.max(osi_values):.4f}")
    print(f"Fraction OSI > 0.3: {np.mean(osi_values > 0.3):.2%}")
    print(f"Fraction OSI > 0.5: {np.mean(osi_values > 0.5):.2%}")
    print(f"Weight stats - Mean: {np.mean(weights[weights > 0]):.4f}, "
          f"Max: {np.max(weights):.4f}")
    print(f"{'='*60}\n")

    return {
        'mean_osi': np.mean(osi_values),
        'median_osi': np.median(osi_values),
        'max_osi': np.max(osi_values),
        'frac_osi_03': np.mean(osi_values > 0.3),
        'frac_osi_05': np.mean(osi_values > 0.5),
    }
