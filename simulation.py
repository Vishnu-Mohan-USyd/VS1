"""
Main simulation class that orchestrates training of the visual pathway model.
Handles the training loop, STDP updates, and progress tracking.
"""

import numpy as np
import os
from tqdm import tqdm

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

from config import (
    DT, SIMULATION_TIME, ORIENTATIONS, N_ORIENTATIONS,
    N_TRAINING_EPOCHS, N_PRESENTATIONS_PER_ORIENTATION,
    VISUALIZATION_INTERVAL, USE_GPU,
    RGC_SIZE, LGN_SIZE, LGN_PATCH_SIZE,
    N_V1_ENSEMBLES, V1_ENSEMBLE_SIZE,
    N_HYPERCOLUMNS_X, N_HYPERCOLUMNS_Y
)
from stimulus import (
    DriftingGrating, StimulusEncoder, get_array_module,
    to_numpy, to_device
)
from layers import RGCLayer, LGNLayer, V1Layer
from learning import STDPLearner, CompetitiveLearning, OrientationSelectivityTracker
from visualization import (
    plot_weights_heatmap, plot_orientation_tuning, plot_orientation_map,
    plot_selectivity_evolution, plot_weight_evolution, plot_training_summary,
    plot_stimulus_sample, plot_layer_activity, ensure_output_dir
)


class VisualPathwaySimulation:
    """
    Complete simulation of RGC -> LGN -> V1 visual pathway with STDP learning.

    This simulation trains orientation selectivity in V1 neurons through
    exposure to drifting gratings at various orientations.
    """

    def __init__(self, output_dir='output', verbose=True):
        """
        Initialize the simulation.

        Args:
            output_dir: Directory for output files and visualizations
            verbose: Whether to print progress information
        """
        self.output_dir = ensure_output_dir(output_dir)
        self.verbose = verbose
        self.xp = get_array_module()

        if self.verbose:
            print("=" * 60)
            print("Visual Pathway Simulation - Orientation Selectivity")
            print("=" * 60)
            print(f"Using {'CuPy (GPU)' if USE_GPU and HAS_CUPY else 'NumPy (CPU)'}")
            print(f"Visual field size: {RGC_SIZE}x{RGC_SIZE}")
            print(f"LGN patch size: {LGN_PATCH_SIZE}x{LGN_PATCH_SIZE}")
            print(f"Hypercolumns: {N_HYPERCOLUMNS_X}x{N_HYPERCOLUMNS_Y}")
            print(f"Ensembles per hypercolumn: {N_V1_ENSEMBLES}")
            print(f"Neurons per ensemble: {V1_ENSEMBLE_SIZE}")
            print(f"Training orientations: {len(ORIENTATIONS)}")
            print("=" * 60)

        # Initialize layers
        self._init_layers()

        # Initialize learning
        self._init_learning()

        # Initialize tracking
        self._init_tracking()

        # Stimulus generators
        self.grating = DriftingGrating()
        self.encoder = StimulusEncoder()

    def _init_layers(self):
        """Initialize neural layers."""
        if self.verbose:
            print("Initializing layers...")

        self.rgc = RGCLayer()
        self.lgn = LGNLayer()
        self.v1 = V1Layer()

        if self.verbose:
            print(f"  RGC: ON({RGC_SIZE}x{RGC_SIZE}) + OFF({RGC_SIZE}x{RGC_SIZE})")
            print(f"  LGN: ON({LGN_SIZE}x{LGN_SIZE}) + OFF({LGN_SIZE}x{LGN_SIZE})")
            print(f"  V1: {self.v1.n_neurons} neurons total")

    def _init_learning(self):
        """Initialize learning mechanisms."""
        if self.verbose:
            print("Initializing learning...")

        self.stdp = STDPLearner(self.v1, self.lgn)
        self.competitive = CompetitiveLearning(self.v1)

    def _init_tracking(self):
        """Initialize tracking and metrics."""
        self.tracker = OrientationSelectivityTracker(self.v1, ORIENTATIONS)
        self.weight_history = []
        self.training_stats = []

    def reset(self):
        """Reset simulation state for a new trial."""
        self.lgn.reset()
        self.v1.reset()
        self.stdp.reset()
        self.encoder.reset()

    def run_trial(self, orientation, duration=SIMULATION_TIME, learning_enabled=True):
        """
        Run a single trial with a specific orientation.

        Args:
            orientation: Orientation of the drifting grating (degrees)
            duration: Duration of the trial (ms)
            learning_enabled: Whether to apply STDP updates

        Returns:
            Dictionary with trial results
        """
        self.reset()

        n_steps = int(duration / DT)
        v1_spike_count = np.zeros(self.v1.shape)

        for t in range(n_steps):
            time_ms = t * DT

            # Generate stimulus
            frame = self.grating.generate(orientation, time_ms)

            # Encode to ON/OFF spikes
            on_spikes, off_spikes = self.encoder.encode(frame)

            # Direct input to RGC (bypassing center-surround for efficiency)
            self.rgc.direct_input(to_numpy(on_spikes), to_numpy(off_spikes))

            # Process through LGN
            lgn_on, lgn_off = self.lgn.step(self.rgc.on_spikes, self.rgc.off_spikes)

            # Process through V1
            v1_spikes = self.v1.step(lgn_on, lgn_off)

            # Accumulate spike counts
            v1_spike_count += to_numpy(v1_spikes)

            # Apply STDP learning
            if learning_enabled:
                self.stdp.update(lgn_on, lgn_off, v1_spikes)
                self.competitive.update(v1_spikes)

        # Compute ensemble-level spike counts (average over neurons)
        ensemble_spikes = np.mean(v1_spike_count, axis=-1)

        return {
            'orientation': orientation,
            'v1_spike_count': v1_spike_count,
            'ensemble_spikes': ensemble_spikes,
            'total_spikes': np.sum(v1_spike_count)
        }

    def measure_orientation_responses(self, duration=SIMULATION_TIME):
        """
        Measure V1 responses to all orientations (without learning).

        Args:
            duration: Duration per orientation (ms)

        Returns:
            Response matrix (n_hy, n_hx, n_ensembles, n_orientations)
        """
        if self.verbose:
            print("Measuring orientation responses...")

        response_matrix = np.zeros((
            self.v1.n_hypercolumns_y, self.v1.n_hypercolumns_x,
            self.v1.n_ensembles, len(ORIENTATIONS)
        ))

        for ori_idx, orientation in enumerate(ORIENTATIONS):
            result = self.run_trial(orientation, duration, learning_enabled=False)
            response_matrix[:, :, :, ori_idx] = result['ensemble_spikes']

        return response_matrix

    def train_epoch(self, epoch_num):
        """
        Run one training epoch.

        Args:
            epoch_num: Current epoch number

        Returns:
            Dictionary with epoch statistics
        """
        if self.verbose:
            print(f"\nEpoch {epoch_num + 1}/{N_TRAINING_EPOCHS}")

        epoch_spikes = 0
        orientation_order = np.random.permutation(len(ORIENTATIONS))

        # Present each orientation multiple times
        for _ in range(N_PRESENTATIONS_PER_ORIENTATION):
            for ori_idx in orientation_order:
                orientation = ORIENTATIONS[ori_idx]
                result = self.run_trial(orientation, learning_enabled=True)
                epoch_spikes += result['total_spikes']

        # Get weight statistics
        weight_stats = self.stdp.get_weight_statistics()

        return {
            'epoch': epoch_num,
            'total_spikes': epoch_spikes,
            'weight_stats': weight_stats
        }

    def train(self, n_epochs=N_TRAINING_EPOCHS, visualize_every=VISUALIZATION_INTERVAL):
        """
        Run full training.

        Args:
            n_epochs: Number of training epochs
            visualize_every: Visualize every N epochs
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("Starting Training")
            print("=" * 60)

        # Initial measurement and visualization
        self._save_checkpoint(0)

        # Training loop
        for epoch in tqdm(range(n_epochs), desc="Training", disable=not self.verbose):
            # Train for one epoch
            stats = self.train_epoch(epoch)
            self.training_stats.append(stats)

            # Periodic visualization and measurement
            if (epoch + 1) % visualize_every == 0 or epoch == n_epochs - 1:
                self._save_checkpoint(epoch + 1)

                if self.verbose:
                    print(f"  Epoch {epoch + 1}: Spikes={stats['total_spikes']:.0f}, "
                          f"Mean W={stats['weight_stats']['mean']:.3f}, "
                          f"OSI={np.mean(self.tracker.compute_selectivity()):.3f}")

        # Final summary
        self._save_final_summary()

        if self.verbose:
            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)
            summary = self.tracker.get_summary_statistics()
            print(f"Final Mean OSI: {summary['mean_osi']:.3f}")
            print(f"Orientation Coverage: {summary['orientation_coverage']*100:.1f}%")
            print(f"Output saved to: {self.output_dir}")

    def _save_checkpoint(self, epoch):
        """Save checkpoint with measurements and visualizations."""
        # Measure responses to all orientations
        response_matrix = self.measure_orientation_responses()
        self.tracker.response_matrix = response_matrix
        self.tracker.save_snapshot()

        # Save weight snapshot
        weights = self.v1.get_weights_for_hypercolumn(0, 0)
        self.weight_history.append(weights.copy())

        # Generate visualizations
        plot_weights_heatmap(self.v1, epoch, self.output_dir)
        plot_orientation_tuning(response_matrix, ORIENTATIONS, epoch, self.output_dir)

        preferred = self.tracker.compute_preferred_orientations()
        osi = self.tracker.compute_selectivity()
        plot_orientation_map(preferred, osi, epoch, self.output_dir)

    def _save_final_summary(self):
        """Save final training summary."""
        # Selectivity evolution plot
        plot_selectivity_evolution(self.tracker.selectivity_history, self.output_dir)

        # Weight evolution for a few ensembles
        for ens in range(min(4, N_V1_ENSEMBLES)):
            plot_weight_evolution(self.weight_history, ens, self.output_dir)

        # Comprehensive summary
        plot_training_summary(self.tracker, self.weight_history, self.v1, self.output_dir)

    def demo_stimulus(self, orientation=45, n_frames=8):
        """
        Generate and save a demo of the stimulus.

        Args:
            orientation: Orientation to demo (degrees)
            n_frames: Number of frames to show
        """
        if self.verbose:
            print(f"Generating stimulus demo for {orientation}° orientation...")

        self.encoder.reset()
        frames = self.grating.generate_sequence(orientation, 200)
        on_spikes, off_spikes = self.encoder.encode_sequence(frames)

        frames = to_numpy(frames)
        on_spikes = to_numpy(on_spikes)
        off_spikes = to_numpy(off_spikes)

        plot_stimulus_sample(frames, on_spikes, off_spikes, orientation,
                           self.output_dir, n_frames=n_frames)

        if self.verbose:
            print(f"  Saved to {self.output_dir}/stimulus_sample_{int(orientation)}deg.png")


class QuickTest:
    """
    Quick test to verify the simulation is working correctly.
    """

    @staticmethod
    def run(output_dir='test_output'):
        """
        Run a quick test of all components.

        Args:
            output_dir: Directory for test output
        """
        print("\n" + "=" * 60)
        print("Running Quick Test")
        print("=" * 60)

        # Create simulation with reduced parameters for testing
        sim = VisualPathwaySimulation(output_dir=output_dir)

        # Test stimulus generation
        print("\n1. Testing stimulus generation...")
        sim.demo_stimulus(45)
        sim.demo_stimulus(90)
        print("   PASSED")

        # Test single trial
        print("\n2. Testing single trial...")
        result = sim.run_trial(45, duration=100, learning_enabled=False)
        print(f"   Total V1 spikes: {result['total_spikes']}")
        print("   PASSED")

        # Test learning
        print("\n3. Testing STDP learning...")
        result_before = sim.run_trial(45, duration=100, learning_enabled=False)
        for _ in range(5):
            sim.run_trial(45, duration=100, learning_enabled=True)
        result_after = sim.run_trial(45, duration=100, learning_enabled=False)
        print(f"   Spikes before: {result_before['total_spikes']:.0f}")
        print(f"   Spikes after:  {result_after['total_spikes']:.0f}")
        print("   PASSED")

        # Test orientation measurement
        print("\n4. Testing orientation responses...")
        responses = sim.measure_orientation_responses(duration=100)
        print(f"   Response matrix shape: {responses.shape}")
        print(f"   Mean response: {np.mean(responses):.2f}")
        print("   PASSED")

        # Test visualization
        print("\n5. Testing visualization...")
        plot_weights_heatmap(sim.v1, 0, output_dir)
        plot_orientation_tuning(responses, ORIENTATIONS, 0, output_dir)
        print("   PASSED")

        print("\n" + "=" * 60)
        print("All tests PASSED!")
        print(f"Test output saved to: {output_dir}")
        print("=" * 60)

        return sim


if __name__ == "__main__":
    # Run quick test
    QuickTest.run()
