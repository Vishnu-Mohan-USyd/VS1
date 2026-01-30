"""
Neural layers for the visual pathway.
Implements RGC, LGN, and V1 (L4) layers with appropriate connectivity.
"""

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

from config import (
    DT, USE_GPU,
    RGC_SIZE, RGC_CENTER_SIGMA, RGC_SURROUND_SIGMA,
    RGC_CENTER_WEIGHT, RGC_SURROUND_WEIGHT, RGC_THRESHOLD,
    LGN_SIZE, LGN_RGC_WEIGHT,
    LGN_PATCH_SIZE, N_V1_ENSEMBLES, V1_ENSEMBLE_SIZE,
    N_HYPERCOLUMNS_X, N_HYPERCOLUMNS_Y, N_HYPERCOLUMNS,
    W_LGN_V1_INIT_MIN, W_LGN_V1_INIT_MAX,
    DELAY_MIN, DELAY_MAX,
    LATERAL_INHIBITION_STRENGTH, LATERAL_INHIBITION_TAU
)
from neurons import LGNNeuronGroup, V1NeuronGroup, SpikeMonitor
from stimulus import get_array_module, to_numpy, to_device


class RGCLayer:
    """
    Retinal Ganglion Cell layer with ON and OFF streams.

    Implements center-surround receptive fields for both ON and OFF pathways.
    ON cells: excited by bright center, inhibited by dark surround
    OFF cells: excited by dark center, inhibited by bright surround
    """

    def __init__(self, size=RGC_SIZE, center_sigma=RGC_CENTER_SIGMA,
                 surround_sigma=RGC_SURROUND_SIGMA):
        """
        Initialize RGC layer.

        Args:
            size: Size of the RGC array (size x size for each ON/OFF)
            center_sigma: Gaussian sigma for center receptive field
            surround_sigma: Gaussian sigma for surround receptive field
        """
        self.xp = get_array_module()
        self.size = size
        self.center_sigma = center_sigma
        self.surround_sigma = surround_sigma

        # Create center-surround filters
        self._create_filters()

        # Output spike arrays
        self.on_spikes = self.xp.zeros((size, size), dtype=self.xp.float32)
        self.off_spikes = self.xp.zeros((size, size), dtype=self.xp.float32)

        # For temporal processing
        self.on_activity = self.xp.zeros((size, size), dtype=self.xp.float32)
        self.off_activity = self.xp.zeros((size, size), dtype=self.xp.float32)

    def _create_filters(self):
        """Create center-surround Gaussian filters."""
        xp = self.xp

        # Determine filter size (should be large enough to cover surround)
        filter_size = int(6 * self.surround_sigma) + 1
        if filter_size % 2 == 0:
            filter_size += 1

        # Create coordinate grid for filter
        x = xp.arange(filter_size) - filter_size // 2
        y = xp.arange(filter_size) - filter_size // 2
        X, Y = xp.meshgrid(x, y)
        r2 = X**2 + Y**2

        # Center Gaussian (excitatory for ON, inhibitory for OFF)
        center = xp.exp(-r2 / (2 * self.center_sigma**2))
        center = center / xp.sum(center)

        # Surround Gaussian (inhibitory for ON, excitatory for OFF)
        surround = xp.exp(-r2 / (2 * self.surround_sigma**2))
        surround = surround / xp.sum(surround)

        # Difference of Gaussians (DoG) filter
        self.dog_filter = (RGC_CENTER_WEIGHT * center -
                          RGC_SURROUND_WEIGHT * surround).astype(xp.float32)

        self.filter_size = filter_size

    def _convolve2d(self, image, kernel):
        """2D convolution with zero-padding."""
        xp = self.xp

        # Pad image
        pad = self.filter_size // 2
        padded = xp.pad(image, pad, mode='constant', constant_values=0)

        # Naive convolution (could be optimized with FFT or im2col)
        h, w = image.shape
        result = xp.zeros_like(image)

        for i in range(h):
            for j in range(w):
                patch = padded[i:i+self.filter_size, j:j+self.filter_size]
                result[i, j] = xp.sum(patch * kernel)

        return result

    def _fast_convolve2d(self, image, kernel):
        """Fast 2D convolution using FFT."""
        xp = self.xp

        # Pad kernel to image size
        h, w = image.shape
        kh, kw = kernel.shape

        # Pad both to avoid circular convolution effects
        ph, pw = h + kh - 1, w + kw - 1

        # Use next power of 2 for efficiency
        fh = int(2 ** np.ceil(np.log2(ph)))
        fw = int(2 ** np.ceil(np.log2(pw)))

        # Pad image and kernel
        image_padded = xp.zeros((fh, fw), dtype=xp.float32)
        image_padded[:h, :w] = image

        kernel_padded = xp.zeros((fh, fw), dtype=xp.float32)
        kernel_padded[:kh, :kw] = kernel

        # FFT convolution
        image_fft = xp.fft.fft2(image_padded)
        kernel_fft = xp.fft.fft2(kernel_padded)
        result_fft = image_fft * kernel_fft
        result = xp.real(xp.fft.ifft2(result_fft))

        # Crop to original size (with offset for center)
        offset_h = (kh - 1) // 2
        offset_w = (kw - 1) // 2
        result = result[offset_h:offset_h+h, offset_w:offset_w+w]

        return result.astype(xp.float32)

    def process(self, luminance_frame):
        """
        Process a luminance frame through ON and OFF channels.

        Args:
            luminance_frame: 2D array of luminance values in range [-1, 1]

        Returns:
            Tuple of (on_spikes, off_spikes) boolean arrays
        """
        xp = self.xp

        # Apply center-surround filtering
        # ON cells respond positively to positive filter output
        # OFF cells respond positively to negative filter output
        filtered = self._fast_convolve2d(luminance_frame, self.dog_filter)

        # ON pathway: positive response to center-bright stimuli
        on_response = xp.maximum(filtered, 0)

        # OFF pathway: positive response to center-dark stimuli
        off_response = xp.maximum(-filtered, 0)

        # Convert to spikes using Poisson process
        # Normalize responses to reasonable firing probabilities
        on_rate = 0.01 + 0.1 * on_response
        off_rate = 0.01 + 0.1 * off_response

        self.on_spikes = (xp.random.random((self.size, self.size)) < on_rate).astype(xp.float32)
        self.off_spikes = (xp.random.random((self.size, self.size)) < off_rate).astype(xp.float32)

        # Store activity for potential visualization
        self.on_activity = on_response
        self.off_activity = off_response

        return self.on_spikes, self.off_spikes

    def direct_input(self, on_spikes, off_spikes):
        """
        Directly set ON and OFF spikes from pre-encoded input.

        Args:
            on_spikes: ON channel spike array
            off_spikes: OFF channel spike array
        """
        self.on_spikes = to_device(on_spikes.astype(np.float32))
        self.off_spikes = to_device(off_spikes.astype(np.float32))


class LGNLayer:
    """
    Lateral Geniculate Nucleus layer.

    Receives input from RGC and maintains retinotopic organization.
    Separate ON and OFF pathways.
    """

    def __init__(self, size=LGN_SIZE, dt=DT):
        """
        Initialize LGN layer.

        Args:
            size: Size of the LGN array (size x size for each ON/OFF)
            dt: Time step (ms)
        """
        self.xp = get_array_module()
        self.size = size
        self.dt = dt

        # LGN neurons for ON and OFF pathways
        self.on_neurons = LGNNeuronGroup((size, size), dt=dt)
        self.off_neurons = LGNNeuronGroup((size, size), dt=dt)

        # Weight from RGC to LGN (one-to-one mapping with gain)
        self.rgc_weight = LGN_RGC_WEIGHT

        # Spike monitors
        self.on_monitor = SpikeMonitor(self.on_neurons)
        self.off_monitor = SpikeMonitor(self.off_neurons)

    def reset(self):
        """Reset LGN state."""
        self.on_neurons.reset()
        self.off_neurons.reset()
        self.on_monitor.reset()
        self.off_monitor.reset()

    def step(self, rgc_on_spikes, rgc_off_spikes):
        """
        Process one timestep of LGN activity.

        Args:
            rgc_on_spikes: ON channel RGC spikes
            rgc_off_spikes: OFF channel RGC spikes

        Returns:
            Tuple of (on_spikes, off_spikes)
        """
        # Convert RGC spikes to input current for LGN
        # Each RGC spike generates a strong current injection (PSP-like)
        on_current = rgc_on_spikes * self.rgc_weight * 8.0  # mV per spike
        off_current = rgc_off_spikes * self.rgc_weight * 8.0

        # Update LGN neurons
        on_spikes = self.on_neurons.step(on_current)
        off_spikes = self.off_neurons.step(off_current)

        # Record spikes
        self.on_monitor.record()
        self.off_monitor.record()

        return on_spikes, off_spikes

    def get_combined_spikes(self):
        """Get combined ON+OFF spikes as a single array."""
        return self.on_neurons.spikes + self.off_neurons.spikes


class V1Layer:
    """
    Primary Visual Cortex Layer 4.

    Organized as hypercolumns (orientation columns), where each hypercolumn
    receives input from a small patch of LGN neurons.

    Each hypercolumn contains multiple ensembles, each of which will
    develop preference for a specific orientation through STDP learning.
    """

    def __init__(self, n_hypercolumns_x=N_HYPERCOLUMNS_X,
                 n_hypercolumns_y=N_HYPERCOLUMNS_Y,
                 n_ensembles=N_V1_ENSEMBLES,
                 ensemble_size=V1_ENSEMBLE_SIZE,
                 patch_size=LGN_PATCH_SIZE,
                 dt=DT):
        """
        Initialize V1 layer.

        Args:
            n_hypercolumns_x: Number of hypercolumns in x direction
            n_hypercolumns_y: Number of hypercolumns in y direction
            n_ensembles: Number of orientation ensembles per hypercolumn
            ensemble_size: Number of neurons per ensemble
            patch_size: Size of LGN patch projecting to each hypercolumn
            dt: Time step (ms)
        """
        self.xp = get_array_module()
        self.n_hypercolumns_x = n_hypercolumns_x
        self.n_hypercolumns_y = n_hypercolumns_y
        self.n_hypercolumns = n_hypercolumns_x * n_hypercolumns_y
        self.n_ensembles = n_ensembles
        self.ensemble_size = ensemble_size
        self.patch_size = patch_size
        self.dt = dt

        # Total neurons in V1
        # Shape: (n_hypercolumns_y, n_hypercolumns_x, n_ensembles, ensemble_size)
        self.shape = (n_hypercolumns_y, n_hypercolumns_x, n_ensembles, ensemble_size)
        self.n_neurons = int(np.prod(self.shape))

        # V1 neurons
        self.neurons = V1NeuronGroup(self.shape, dt=dt)

        # Spike monitor
        self.monitor = SpikeMonitor(self.neurons)

        # Initialize LGN -> V1 weights
        # For each hypercolumn, we have n_ensembles projections from the LGN patch
        # Each projection has different initial weights and delays
        self._initialize_weights()

        # Lateral inhibition between ensembles within each hypercolumn
        self.lateral_inhibition = self.xp.zeros(
            (n_hypercolumns_y, n_hypercolumns_x, n_ensembles),
            dtype=self.xp.float32
        )

        # Trace for lateral inhibition dynamics
        self.lateral_trace = self.xp.zeros_like(self.lateral_inhibition)

    def _initialize_weights(self):
        """Initialize synaptic weights from LGN patches to V1 ensembles."""
        xp = self.xp

        # Weight shape: (n_hypercolumns_y, n_hypercolumns_x, n_ensembles,
        #                ensemble_size, 2, patch_size, patch_size)
        # The '2' dimension is for ON and OFF channels
        weight_shape = (
            self.n_hypercolumns_y, self.n_hypercolumns_x,
            self.n_ensembles, self.ensemble_size,
            2, self.patch_size, self.patch_size
        )

        # Initialize weights with random values
        # Different initial weights for each ensemble to break symmetry
        self.weights = xp.random.uniform(
            W_LGN_V1_INIT_MIN, W_LGN_V1_INIT_MAX,
            weight_shape
        ).astype(xp.float32)

        # Add orientation bias to initial weights to help break symmetry
        # Each ensemble gets a slight bias toward a particular orientation
        self._add_initial_orientation_bias()

        # Initialize conduction delays
        # Shape: (n_hypercolumns_y, n_hypercolumns_x, n_ensembles,
        #         ensemble_size, 2, patch_size, patch_size)
        self.delays = xp.random.uniform(
            DELAY_MIN, DELAY_MAX, weight_shape
        ).astype(xp.float32)

        # Delay buffers for each hypercolumn
        max_delay_steps = int(np.ceil(DELAY_MAX / self.dt)) + 1
        self.max_delay_steps = max_delay_steps

        # Buffer shape: (max_delay_steps, n_hypercolumns_y, n_hypercolumns_x,
        #                2, patch_size, patch_size)
        self.delay_buffer = xp.zeros(
            (max_delay_steps, self.n_hypercolumns_y, self.n_hypercolumns_x,
             2, self.patch_size, self.patch_size),
            dtype=xp.float32
        )
        self.buffer_idx = 0

    def _add_initial_orientation_bias(self):
        """Add slight orientation bias to initial weights."""
        xp = self.xp

        for ens_idx in range(self.n_ensembles):
            # Each ensemble gets a preferred orientation
            preferred_ori = ens_idx * 180.0 / self.n_ensembles
            theta = xp.deg2rad(preferred_ori)

            # Create a slight elongation in the weights along preferred orientation
            x = xp.arange(self.patch_size) - self.patch_size / 2
            y = xp.arange(self.patch_size) - self.patch_size / 2
            X, Y = xp.meshgrid(x, y)

            # Rotate coordinates
            X_rot = X * xp.cos(theta) + Y * xp.sin(theta)

            # Create orientation-tuned modulation
            modulation = 1.0 + 0.1 * xp.cos(2 * xp.pi * X_rot / self.patch_size)

            # Apply to weights for this ensemble
            for hy in range(self.n_hypercolumns_y):
                for hx in range(self.n_hypercolumns_x):
                    for n in range(self.ensemble_size):
                        for c in range(2):  # ON and OFF
                            self.weights[hy, hx, ens_idx, n, c] *= modulation

    def reset(self):
        """Reset V1 state."""
        self.neurons.reset()
        self.monitor.reset()
        self.lateral_inhibition.fill(0)
        self.lateral_trace.fill(0)
        self.delay_buffer.fill(0)
        self.buffer_idx = 0

    def step(self, lgn_on_spikes, lgn_off_spikes):
        """
        Process one timestep of V1 activity.

        Args:
            lgn_on_spikes: LGN ON channel spikes (size x size)
            lgn_off_spikes: LGN OFF channel spikes (size x size)

        Returns:
            V1 spikes array
        """
        xp = self.xp

        # Update delay buffer with current LGN spikes
        self._update_delay_buffer(lgn_on_spikes, lgn_off_spikes)

        # Compute input current from delayed LGN spikes
        input_current = self._compute_input_current()

        # Apply lateral inhibition
        input_current = self._apply_lateral_inhibition(input_current)

        # Update V1 neurons
        spikes = self.neurons.step(input_current)

        # Record spikes
        self.monitor.record()

        # Update lateral inhibition based on current activity
        self._update_lateral_inhibition(spikes)

        return spikes

    def _update_delay_buffer(self, lgn_on_spikes, lgn_off_spikes):
        """Update the delay buffer with current LGN spikes."""
        xp = self.xp

        # Extract patches from LGN for each hypercolumn
        for hy in range(self.n_hypercolumns_y):
            for hx in range(self.n_hypercolumns_x):
                # Calculate patch coordinates
                y_start = hy * self.patch_size
                x_start = hx * self.patch_size

                # Extract ON and OFF patches
                on_patch = lgn_on_spikes[y_start:y_start+self.patch_size,
                                         x_start:x_start+self.patch_size]
                off_patch = lgn_off_spikes[y_start:y_start+self.patch_size,
                                           x_start:x_start+self.patch_size]

                # Store in buffer
                self.delay_buffer[self.buffer_idx, hy, hx, 0] = on_patch
                self.delay_buffer[self.buffer_idx, hy, hx, 1] = off_patch

        # Advance buffer index
        self.buffer_idx = (self.buffer_idx + 1) % self.max_delay_steps

    def _compute_input_current(self):
        """Compute input current to V1 neurons from delayed LGN spikes.

        Uses vectorized operations for speed. Simplifies delay model to use
        average delay per ensemble for computational efficiency.
        """
        xp = self.xp

        # Output current shape matches V1 neuron shape
        current = xp.zeros(self.shape, dtype=xp.float32)

        # Use a simplified delay model: average delay per ensemble
        # This allows vectorized computation across all synapses
        for hy in range(self.n_hypercolumns_y):
            for hx in range(self.n_hypercolumns_x):
                # Get the spikes from delay buffer using average delay
                # Use most recent buffer entry for simplicity (delay ≈ 1 timestep)
                buf_idx = (self.buffer_idx - 2) % self.max_delay_steps

                # Get ON and OFF patches from buffer
                on_spikes = self.delay_buffer[buf_idx, hy, hx, 0]  # (patch_y, patch_x)
                off_spikes = self.delay_buffer[buf_idx, hy, hx, 1]

                # Stack for vectorized computation
                lgn_spikes = xp.stack([on_spikes, off_spikes], axis=0)  # (2, patch_y, patch_x)

                for ens in range(self.n_ensembles):
                    # Get weights for all neurons in this ensemble
                    # Shape: (ensemble_size, 2, patch_y, patch_x)
                    weights = self.weights[hy, hx, ens]

                    # Compute weighted sum for all neurons at once
                    # weights: (ensemble_size, 2, patch_y, patch_x)
                    # lgn_spikes: (2, patch_y, patch_x)
                    # Result: (ensemble_size,)
                    weighted_input = xp.sum(weights * lgn_spikes, axis=(1, 2, 3))

                    # Scale and assign to current
                    current[hy, hx, ens, :] = weighted_input * 12.0

        return current

    def _apply_lateral_inhibition(self, input_current):
        """Apply lateral inhibition between ensembles (vectorized)."""
        xp = self.xp

        # Sum of all lateral traces per hypercolumn
        total_trace = xp.sum(self.lateral_trace, axis=-1, keepdims=True)  # (hy, hx, 1)

        # Each ensemble is inhibited by sum of others = total - self
        inhibition = total_trace - self.lateral_trace  # (hy, hx, n_ensembles)

        # Apply inhibition to all neurons in each ensemble
        inhibition_expanded = inhibition[:, :, :, xp.newaxis]  # (hy, hx, n_ensembles, 1)
        input_current -= LATERAL_INHIBITION_STRENGTH * inhibition_expanded

        return input_current

    def _update_lateral_inhibition(self, spikes):
        """Update lateral inhibition trace based on spiking activity."""
        xp = self.xp

        # Decay existing trace
        decay = xp.exp(-self.dt / LATERAL_INHIBITION_TAU)
        self.lateral_trace *= decay

        # Add new spikes to trace
        # Sum spikes across neurons within each ensemble
        ensemble_activity = xp.sum(spikes, axis=-1)  # (hy, hx, n_ensembles)
        self.lateral_trace += ensemble_activity

    def get_ensemble_activity(self):
        """Get firing rate of each ensemble (averaged over neurons)."""
        # Get spike counts from monitor
        rates = self.monitor.get_firing_rates()
        rates = rates.reshape(self.shape)

        # Average over neurons within each ensemble
        ensemble_rates = np.mean(rates, axis=-1)  # (hy, hx, n_ensembles)

        return ensemble_rates

    def get_weights_for_hypercolumn(self, hy, hx):
        """
        Get weights for a specific hypercolumn.

        Returns:
            Array of shape (n_ensembles, ensemble_size, 2, patch_size, patch_size)
        """
        return to_numpy(self.weights[hy, hx])

    def get_preferred_orientations(self, orientations):
        """
        Determine preferred orientation for each ensemble based on responses.

        Args:
            orientations: Array of orientations that were presented

        Returns:
            Array of preferred orientations (n_hypercolumns_y, n_hypercolumns_x, n_ensembles)
        """
        # This requires running the network on each orientation and recording responses
        # For now, return placeholder based on initial bias
        preferred = np.zeros((self.n_hypercolumns_y, self.n_hypercolumns_x, self.n_ensembles))
        for ens in range(self.n_ensembles):
            preferred[:, :, ens] = ens * 180.0 / self.n_ensembles

        return preferred


if __name__ == "__main__":
    # Test layers
    import matplotlib.pyplot as plt
    from stimulus import DriftingGrating, StimulusEncoder, to_numpy

    print("Testing visual pathway layers...")

    # Create layers
    rgc = RGCLayer()
    lgn = LGNLayer()
    v1 = V1Layer()

    # Create stimulus
    grating = DriftingGrating()
    encoder = StimulusEncoder()

    # Generate test stimulus
    orientation = 45  # degrees
    duration = 200  # ms

    print(f"Testing with {orientation} degree grating for {duration} ms...")

    # Reset layers
    lgn.reset()
    v1.reset()

    # Run simulation
    n_steps = int(duration / DT)
    v1_activity = []

    for t in range(n_steps):
        time_ms = t * DT

        # Generate stimulus frame
        frame = grating.generate(orientation, time_ms)

        # Encode to spikes
        on_spikes, off_spikes = encoder.encode(frame)

        # Process through RGC (or use direct encoded spikes)
        rgc.direct_input(to_numpy(on_spikes), to_numpy(off_spikes))

        # Process through LGN
        lgn_on, lgn_off = lgn.step(rgc.on_spikes, rgc.off_spikes)

        # Process through V1
        v1_spikes = v1.step(lgn_on, lgn_off)

        # Record V1 activity
        v1_activity.append(to_numpy(v1_spikes).copy())

    # Analyze results
    v1_activity = np.array(v1_activity)
    total_spikes = np.sum(v1_activity)

    print(f"\nResults:")
    print(f"Total V1 spikes: {total_spikes}")
    print(f"V1 activity shape: {v1_activity.shape}")

    # Get ensemble activity
    ensemble_activity = v1.get_ensemble_activity()
    print(f"Ensemble activity shape: {ensemble_activity.shape}")
    print(f"Mean ensemble activity: {np.mean(ensemble_activity):.4f} Hz")

    # Plot weights for one hypercolumn
    weights = v1.get_weights_for_hypercolumn(0, 0)

    fig, axes = plt.subplots(2, N_V1_ENSEMBLES, figsize=(16, 4))
    for ens in range(N_V1_ENSEMBLES):
        # Average over neurons, show ON channel
        w_on = np.mean(weights[ens, :, 0], axis=0)
        w_off = np.mean(weights[ens, :, 1], axis=0)

        axes[0, ens].imshow(w_on, cmap='Reds')
        axes[0, ens].set_title(f'Ens {ens} ON')
        axes[0, ens].axis('off')

        axes[1, ens].imshow(w_off, cmap='Blues')
        axes[1, ens].set_title(f'Ens {ens} OFF')
        axes[1, ens].axis('off')

    plt.suptitle('Initial LGN->V1 Weights (Hypercolumn 0,0)')
    plt.tight_layout()
    plt.savefig('layer_test_weights.png', dpi=100)
    plt.close()

    print("\nSaved weight visualization to layer_test_weights.png")
