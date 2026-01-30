"""
Visualization utilities for monitoring training and debugging.
Provides various plots for weight evolution, orientation selectivity, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import os

from config import N_V1_ENSEMBLES, LGN_PATCH_SIZE


def ensure_output_dir(output_dir='output'):
    """Ensure output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def plot_weights_heatmap(v1_layer, epoch, output_dir='output', hypercolumn=(0, 0)):
    """
    Plot weight heatmaps for all ensembles in a hypercolumn.

    Args:
        v1_layer: V1 layer with weights
        epoch: Current training epoch
        output_dir: Directory to save plots
        hypercolumn: Tuple (hy, hx) specifying which hypercolumn to plot
    """
    ensure_output_dir(output_dir)

    hy, hx = hypercolumn
    weights = v1_layer.get_weights_for_hypercolumn(hy, hx)
    n_ensembles = weights.shape[0]

    # Create figure with ON and OFF weights for each ensemble
    fig, axes = plt.subplots(3, n_ensembles, figsize=(2*n_ensembles, 6))

    for ens in range(n_ensembles):
        # Average over neurons in ensemble
        w_on = np.mean(weights[ens, :, 0], axis=0)  # ON channel
        w_off = np.mean(weights[ens, :, 1], axis=0)  # OFF channel
        w_combined = w_on - w_off  # ON - OFF (like receptive field)

        # ON weights
        im0 = axes[0, ens].imshow(w_on, cmap='Reds', vmin=0, vmax=1)
        axes[0, ens].set_title(f'E{ens}', fontsize=10)
        axes[0, ens].axis('off')

        # OFF weights
        im1 = axes[1, ens].imshow(w_off, cmap='Blues', vmin=0, vmax=1)
        axes[1, ens].axis('off')

        # Combined (ON - OFF)
        im2 = axes[2, ens].imshow(w_combined, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2, ens].axis('off')

    # Row labels
    axes[0, 0].set_ylabel('ON', fontsize=12)
    axes[1, 0].set_ylabel('OFF', fontsize=12)
    axes[2, 0].set_ylabel('ON-OFF', fontsize=12)

    # Add colorbars
    fig.colorbar(im0, ax=axes[0, :], shrink=0.6, label='Weight')
    fig.colorbar(im1, ax=axes[1, :], shrink=0.6, label='Weight')
    fig.colorbar(im2, ax=axes[2, :], shrink=0.6, label='ON-OFF')

    plt.suptitle(f'LGN→V1 Weights - Hypercolumn ({hy},{hx}) - Epoch {epoch}', fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'weights_heatmap_epoch{epoch:03d}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def plot_orientation_tuning(response_matrix, orientations, epoch, output_dir='output',
                           hypercolumn=(0, 0)):
    """
    Plot orientation tuning curves for all ensembles.

    Args:
        response_matrix: Response matrix (hy, hx, n_ensembles, n_orientations)
        orientations: Array of orientation values
        epoch: Current training epoch
        output_dir: Directory to save plots
        hypercolumn: Which hypercolumn to plot
    """
    ensure_output_dir(output_dir)

    hy, hx = hypercolumn
    responses = response_matrix[hy, hx]  # (n_ensembles, n_orientations)
    n_ensembles = responses.shape[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, n_ensembles))

    for ens in range(n_ensembles):
        # Normalize response
        resp = responses[ens]
        if np.max(resp) > 0:
            resp_norm = resp / np.max(resp)
        else:
            resp_norm = resp

        ax.plot(orientations, resp_norm, 'o-', color=colors[ens],
                label=f'Ensemble {ens}', linewidth=2, markersize=6)

    ax.set_xlabel('Orientation (degrees)', fontsize=12)
    ax.set_ylabel('Normalized Response', fontsize=12)
    ax.set_title(f'Orientation Tuning Curves - Hypercolumn ({hy},{hx}) - Epoch {epoch}', fontsize=14)
    ax.legend(loc='upper right', ncol=2)
    ax.set_xlim([0, 180])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, f'tuning_curves_epoch{epoch:03d}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def plot_orientation_map(preferred_orientations, osi, epoch, output_dir='output'):
    """
    Plot orientation preference map (like pinwheel maps in V1).

    Args:
        preferred_orientations: Array of preferred orientations (hy, hx, n_ensembles)
        osi: Orientation selectivity index (hy, hx, n_ensembles)
        epoch: Current training epoch
        output_dir: Directory to save plots
    """
    ensure_output_dir(output_dir)

    n_hy, n_hx, n_ensembles = preferred_orientations.shape

    # Create a circular plot for each hypercolumn
    fig, axes = plt.subplots(n_hy, n_hx, figsize=(3*n_hx, 3*n_hy))

    if n_hy == 1 and n_hx == 1:
        axes = np.array([[axes]])
    elif n_hy == 1:
        axes = axes.reshape(1, -1)
    elif n_hx == 1:
        axes = axes.reshape(-1, 1)

    # Orientation colormap (HSV-like for orientation)
    cmap = plt.cm.hsv

    for hy in range(n_hy):
        for hx in range(n_hx):
            ax = axes[hy, hx]
            ax.set_aspect('equal')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)

            # Draw ensembles as pie slices around a circle
            patches = []
            colors = []

            for ens in range(n_ensembles):
                # Position on circle
                angle = 2 * np.pi * ens / n_ensembles
                x = np.cos(angle)
                y = np.sin(angle)

                # Color based on preferred orientation
                ori = preferred_orientations[hy, hx, ens]
                color = cmap(ori / 180.0)

                # Size based on OSI
                selectivity = osi[hy, hx, ens]
                radius = 0.3 + 0.2 * selectivity

                circle = Circle((x, y), radius, alpha=0.8)
                patches.append(circle)
                colors.append(color)

                # Add text label
                ax.text(x, y, f'{int(ori)}°', ha='center', va='center',
                       fontsize=8, fontweight='bold')

            collection = PatchCollection(patches, facecolors=colors, edgecolors='black')
            ax.add_collection(collection)

            ax.set_title(f'HC({hy},{hx})', fontsize=10)
            ax.axis('off')

    # Add colorbar for orientation
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 180))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label='Preferred Orientation (°)')

    plt.suptitle(f'Orientation Map - Epoch {epoch}', fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'orientation_map_epoch{epoch:03d}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def plot_selectivity_evolution(selectivity_history, output_dir='output'):
    """
    Plot how orientation selectivity evolves over training.

    Args:
        selectivity_history: List of dictionaries with OSI snapshots
        output_dir: Directory to save plots
    """
    ensure_output_dir(output_dir)

    epochs = range(len(selectivity_history))
    mean_osi = [np.mean(h['osi']) for h in selectivity_history]
    std_osi = [np.std(h['osi']) for h in selectivity_history]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, mean_osi, 'b-', linewidth=2, label='Mean OSI')
    ax.fill_between(epochs,
                   [m - s for m, s in zip(mean_osi, std_osi)],
                   [m + s for m, s in zip(mean_osi, std_osi)],
                   alpha=0.3, color='blue')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Orientation Selectivity Index (OSI)', fontsize=12)
    ax.set_title('Evolution of Orientation Selectivity During Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    filepath = os.path.join(output_dir, 'selectivity_evolution.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def plot_weight_evolution(weight_history, ensemble_idx=0, output_dir='output'):
    """
    Plot how weights evolve for a specific ensemble.

    Args:
        weight_history: List of weight snapshots
        ensemble_idx: Which ensemble to visualize
        output_dir: Directory to save plots
    """
    ensure_output_dir(output_dir)

    n_epochs = len(weight_history)

    # Select epochs to display (up to 10)
    if n_epochs <= 10:
        display_epochs = list(range(n_epochs))
    else:
        display_epochs = np.linspace(0, n_epochs-1, 10, dtype=int).tolist()

    n_display = len(display_epochs)

    fig, axes = plt.subplots(2, n_display, figsize=(2*n_display, 4))

    for i, epoch in enumerate(display_epochs):
        weights = weight_history[epoch]  # (n_ensembles, ensemble_size, 2, patch_y, patch_x)

        # Average over neurons
        w_on = np.mean(weights[ensemble_idx, :, 0], axis=0)
        w_off = np.mean(weights[ensemble_idx, :, 1], axis=0)

        axes[0, i].imshow(w_on, cmap='Reds', vmin=0, vmax=1)
        axes[0, i].set_title(f'E{epoch}', fontsize=9)
        axes[0, i].axis('off')

        axes[1, i].imshow(w_off, cmap='Blues', vmin=0, vmax=1)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('ON', fontsize=10)
    axes[1, 0].set_ylabel('OFF', fontsize=10)

    plt.suptitle(f'Weight Evolution - Ensemble {ensemble_idx}', fontsize=12)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'weight_evolution_ens{ensemble_idx}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def plot_raster(spike_monitor, title='Spike Raster', output_dir='output', filename='raster.png'):
    """
    Plot spike raster from a spike monitor.

    Args:
        spike_monitor: SpikeMonitor object
        title: Plot title
        output_dir: Directory to save plots
        filename: Output filename
    """
    ensure_output_dir(output_dir)

    data = spike_monitor.get_data()
    times = data['times']
    indices = data['indices']

    fig, ax = plt.subplots(figsize=(12, 6))

    if len(times) > 0:
        ax.scatter(times, indices, s=1, alpha=0.5, c='black')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron Index', fontsize=12)
    ax.set_title(title, fontsize=14)

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def plot_training_summary(tracker, weight_history, v1_layer, output_dir='output'):
    """
    Create a comprehensive training summary plot.

    Args:
        tracker: OrientationSelectivityTracker
        weight_history: List of weight snapshots
        v1_layer: V1 layer
        output_dir: Directory to save plots
    """
    ensure_output_dir(output_dir)

    fig = plt.figure(figsize=(16, 12))

    # 1. OSI evolution
    ax1 = fig.add_subplot(2, 3, 1)
    if len(tracker.selectivity_history) > 0:
        epochs = range(len(tracker.selectivity_history))
        mean_osi = [np.mean(h['osi']) for h in tracker.selectivity_history]
        ax1.plot(epochs, mean_osi, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean OSI')
        ax1.set_title('Orientation Selectivity Evolution')
        ax1.grid(True, alpha=0.3)

    # 2. Final tuning curves for one hypercolumn
    ax2 = fig.add_subplot(2, 3, 2)
    responses = tracker.response_matrix[0, 0]  # First hypercolumn
    orientations = tracker.orientations
    n_ensembles = responses.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_ensembles))

    for ens in range(n_ensembles):
        resp = responses[ens]
        if np.max(resp) > 0:
            resp_norm = resp / np.max(resp)
        else:
            resp_norm = resp
        ax2.plot(orientations, resp_norm, 'o-', color=colors[ens], linewidth=1.5)

    ax2.set_xlabel('Orientation (°)')
    ax2.set_ylabel('Normalized Response')
    ax2.set_title('Final Tuning Curves (HC 0,0)')
    ax2.grid(True, alpha=0.3)

    # 3. Preferred orientation distribution
    ax3 = fig.add_subplot(2, 3, 3)
    pref = tracker.compute_preferred_orientations().flatten()
    ax3.hist(pref, bins=len(orientations), range=(0, 180), edgecolor='black')
    ax3.set_xlabel('Preferred Orientation (°)')
    ax3.set_ylabel('Count')
    ax3.set_title('Preferred Orientation Distribution')

    # 4. OSI distribution
    ax4 = fig.add_subplot(2, 3, 4)
    osi = tracker.compute_selectivity().flatten()
    ax4.hist(osi, bins=20, range=(0, 1), edgecolor='black')
    ax4.set_xlabel('OSI')
    ax4.set_ylabel('Count')
    ax4.set_title('OSI Distribution')

    # 5. Initial vs Final weights for one ensemble
    ax5 = fig.add_subplot(2, 3, 5)
    if len(weight_history) >= 2:
        w_init = np.mean(weight_history[0][0, :, 0], axis=0) - np.mean(weight_history[0][0, :, 1], axis=0)
        w_final = np.mean(weight_history[-1][0, :, 0], axis=0) - np.mean(weight_history[-1][0, :, 1], axis=0)

        im = ax5.imshow(np.hstack([w_init, np.zeros((w_init.shape[0], 1)), w_final]),
                       cmap='RdBu_r', vmin=-1, vmax=1)
        ax5.set_title('Initial vs Final Weights (Ens 0)')
        ax5.set_xticks([w_init.shape[1]//2, w_init.shape[1] + 1 + w_final.shape[1]//2])
        ax5.set_xticklabels(['Initial', 'Final'])
        ax5.set_yticks([])
        plt.colorbar(im, ax=ax5)

    # 6. Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    stats = tracker.get_summary_statistics()

    text = f"""Training Summary
    ─────────────────────
    Mean OSI: {stats['mean_osi']:.3f}
    Std OSI: {stats['std_osi']:.3f}

    Mean Tuning Width: {stats['mean_tuning_width']:.1f}°

    Orientation Coverage: {stats['orientation_coverage']*100:.1f}%

    Total Epochs: {len(weight_history)}
    """
    ax6.text(0.1, 0.5, text, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax6.transAxes)

    plt.suptitle('Training Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'training_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def plot_stimulus_sample(frames, on_spikes, off_spikes, orientation,
                        output_dir='output', n_frames=4):
    """
    Plot sample stimulus frames with corresponding ON/OFF spikes.

    Args:
        frames: Raw stimulus frames
        on_spikes: ON channel spikes
        off_spikes: OFF channel spikes
        orientation: Orientation of the stimulus
        output_dir: Directory to save plots
        n_frames: Number of frames to display
    """
    ensure_output_dir(output_dir)

    fig, axes = plt.subplots(3, n_frames, figsize=(3*n_frames, 9))

    for i in range(n_frames):
        idx = i * (len(frames) // n_frames)

        axes[0, i].imshow(frames[idx], cmap='gray', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Frame {idx}')
        axes[0, i].axis('off')

        axes[1, i].imshow(on_spikes[idx], cmap='Reds', vmin=0, vmax=1)
        axes[1, i].axis('off')

        axes[2, i].imshow(off_spikes[idx], cmap='Blues', vmin=0, vmax=1)
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Stimulus', fontsize=12)
    axes[1, 0].set_ylabel('ON spikes', fontsize=12)
    axes[2, 0].set_ylabel('OFF spikes', fontsize=12)

    plt.suptitle(f'Drifting Grating - Orientation: {orientation}°', fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'stimulus_sample_{int(orientation)}deg.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def plot_layer_activity(lgn_layer, v1_layer, time_window, output_dir='output', filename='activity.png'):
    """
    Plot activity heatmaps for LGN and V1 layers.

    Args:
        lgn_layer: LGN layer
        v1_layer: V1 layer
        time_window: Tuple (start, end) for time window
        output_dir: Directory to save plots
        filename: Output filename
    """
    ensure_output_dir(output_dir)

    # Get firing rates
    lgn_on_rates = lgn_layer.on_monitor.get_firing_rates(time_window)
    lgn_off_rates = lgn_layer.off_monitor.get_firing_rates(time_window)
    v1_rates = v1_layer.monitor.get_firing_rates(time_window)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # LGN ON
    im0 = axes[0, 0].imshow(lgn_on_rates, cmap='hot')
    axes[0, 0].set_title('LGN ON Firing Rates')
    plt.colorbar(im0, ax=axes[0, 0], label='Hz')

    # LGN OFF
    im1 = axes[0, 1].imshow(lgn_off_rates, cmap='hot')
    axes[0, 1].set_title('LGN OFF Firing Rates')
    plt.colorbar(im1, ax=axes[0, 1], label='Hz')

    # V1 - average over neurons per ensemble
    v1_ensemble_rates = np.mean(v1_rates.reshape(v1_layer.shape), axis=-1)

    # Flatten ensembles for visualization
    v1_flat = v1_ensemble_rates.reshape(
        v1_layer.n_hypercolumns_y * v1_layer.n_ensembles,
        v1_layer.n_hypercolumns_x
    )

    im2 = axes[1, 0].imshow(v1_flat, cmap='hot', aspect='auto')
    axes[1, 0].set_title('V1 Ensemble Firing Rates')
    axes[1, 0].set_xlabel('Hypercolumn X')
    axes[1, 0].set_ylabel('Hypercolumn Y × Ensemble')
    plt.colorbar(im2, ax=axes[1, 0], label='Hz')

    # V1 ensemble comparison for first hypercolumn
    ensemble_rates = v1_ensemble_rates[0, 0]  # First hypercolumn
    axes[1, 1].bar(range(len(ensemble_rates)), ensemble_rates)
    axes[1, 1].set_xlabel('Ensemble')
    axes[1, 1].set_ylabel('Firing Rate (Hz)')
    axes[1, 1].set_title('V1 Ensemble Rates (HC 0,0)')

    plt.suptitle(f'Layer Activity ({time_window[0]:.0f}-{time_window[1]:.0f} ms)', fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")

    output_dir = ensure_output_dir('test_output')

    # Create dummy data
    n_hy, n_hx, n_ens = 2, 2, 8
    n_ori = 8
    orientations = np.linspace(0, 180, n_ori, endpoint=False)

    # Dummy response matrix
    response_matrix = np.random.rand(n_hy, n_hx, n_ens, n_ori)

    # Make some ensembles have orientation preference
    for ens in range(n_ens):
        preferred_ori_idx = ens % n_ori
        response_matrix[:, :, ens, preferred_ori_idx] += 2.0

    # Compute preferred orientations and OSI
    pref_idx = np.argmax(response_matrix, axis=-1)
    preferred_orientations = orientations[pref_idx]

    # Compute OSI
    r_pref = np.max(response_matrix, axis=-1)
    orth_idx = (pref_idx + n_ori // 2) % n_ori
    hy_idx, hx_idx, ens_idx = np.indices(pref_idx.shape)
    r_orth = response_matrix[hy_idx, hx_idx, ens_idx, orth_idx]
    osi = (r_pref - r_orth) / (r_pref + r_orth + 1e-8)

    # Test plots
    plot_orientation_tuning(response_matrix, orientations, 0, output_dir)
    plot_orientation_map(preferred_orientations, osi, 0, output_dir)

    print(f"Test plots saved to {output_dir}/")
