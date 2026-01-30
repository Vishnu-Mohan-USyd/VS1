"""
Spiking neuron models.
Implements Leaky Integrate-and-Fire (LIF) neurons with GPU support.
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
    LGN_V_REST, LGN_V_RESET, LGN_V_THRESHOLD, LGN_TAU_M, LGN_REFRACTORY,
    V1_V_REST, V1_V_RESET, V1_V_THRESHOLD, V1_TAU_M, V1_REFRACTORY
)
from stimulus import get_array_module, to_numpy, to_device


class LIFNeuronGroup:
    """
    Leaky Integrate-and-Fire neuron group.

    Supports arbitrary neuron arrangements with vectorized computation.
    """

    def __init__(self, shape, v_rest=-70.0, v_reset=-75.0, v_threshold=-55.0,
                 tau_m=20.0, refractory_period=2.0, dt=DT):
        """
        Initialize a group of LIF neurons.

        Args:
            shape: Shape of the neuron array (can be 1D, 2D, etc.)
            v_rest: Resting potential (mV)
            v_reset: Reset potential after spike (mV)
            v_threshold: Spike threshold (mV)
            tau_m: Membrane time constant (ms)
            refractory_period: Refractory period (ms)
            dt: Time step (ms)
        """
        self.xp = get_array_module()
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.n_neurons = int(np.prod(self.shape))

        # Neuron parameters
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.tau_m = tau_m
        self.refractory_period = refractory_period
        self.dt = dt

        # Decay factor for membrane potential
        self.decay = self.xp.exp(-dt / tau_m)

        # State variables
        self.v = self.xp.ones(self.shape, dtype=self.xp.float32) * v_rest
        self.refractory_time = self.xp.zeros(self.shape, dtype=self.xp.float32)
        self.spikes = self.xp.zeros(self.shape, dtype=self.xp.float32)

        # Spike history for STDP
        self.spike_times = self.xp.full(self.shape, -1000.0, dtype=self.xp.float32)
        self.current_time = 0.0

    def reset(self):
        """Reset all neurons to resting state."""
        self.v.fill(self.v_rest)
        self.refractory_time.fill(0.0)
        self.spikes.fill(0.0)
        self.spike_times.fill(-1000.0)
        self.current_time = 0.0

    def step(self, input_current):
        """
        Simulate one time step.

        Args:
            input_current: Input current to each neuron (same shape as neuron array)

        Returns:
            Boolean array of spikes
        """
        xp = self.xp

        # Update current time
        self.current_time += self.dt

        # Ensure input is on correct device
        if isinstance(input_current, np.ndarray) and xp == cp:
            input_current = xp.asarray(input_current)

        # Decrease refractory time
        self.refractory_time = xp.maximum(0, self.refractory_time - self.dt)

        # Find neurons not in refractory period
        active = self.refractory_time <= 0

        # Update membrane potential with leaky integration
        # dV/dt = -(V - V_rest) / tau_m + I / C
        # Using exponential decay for leak, additive current injection
        # V(t+dt) = V_rest + (V(t) - V_rest) * exp(-dt/tau) + I * dt / tau
        # The current is treated as a direct voltage jump (like PSP)
        self.v = xp.where(
            active,
            self.v_rest + (self.v - self.v_rest) * self.decay + input_current,
            self.v
        )

        # Check for spikes
        self.spikes = (self.v >= self.v_threshold).astype(xp.float32)

        # Reset spiking neurons
        spiked = self.spikes > 0
        self.v = xp.where(spiked, self.v_reset, self.v)
        self.refractory_time = xp.where(spiked, self.refractory_period, self.refractory_time)

        # Record spike times
        self.spike_times = xp.where(spiked, self.current_time, self.spike_times)

        return self.spikes

    def get_state(self):
        """Get current state as numpy arrays."""
        return {
            'v': to_numpy(self.v),
            'spikes': to_numpy(self.spikes),
            'spike_times': to_numpy(self.spike_times),
            'current_time': self.current_time
        }


class LGNNeuronGroup(LIFNeuronGroup):
    """LGN neuron group with default LGN parameters."""

    def __init__(self, shape, dt=DT):
        super().__init__(
            shape,
            v_rest=LGN_V_REST,
            v_reset=LGN_V_RESET,
            v_threshold=LGN_V_THRESHOLD,
            tau_m=LGN_TAU_M,
            refractory_period=LGN_REFRACTORY,
            dt=dt
        )


class V1NeuronGroup(LIFNeuronGroup):
    """V1 neuron group with default V1 parameters."""

    def __init__(self, shape, dt=DT):
        super().__init__(
            shape,
            v_rest=V1_V_REST,
            v_reset=V1_V_RESET,
            v_threshold=V1_V_THRESHOLD,
            tau_m=V1_TAU_M,
            refractory_period=V1_REFRACTORY,
            dt=dt
        )


class DelayedSynapseGroup:
    """
    Synapse group with axonal conduction delays.

    Implements a delay line for spike transmission.
    """

    def __init__(self, pre_shape, post_shape, delays, weights, dt=DT):
        """
        Initialize delayed synapse group.

        Args:
            pre_shape: Shape of presynaptic neuron array
            post_shape: Shape of postsynaptic neuron array
            delays: Array of delays (ms) for each synapse
            weights: Array of synaptic weights
            dt: Time step (ms)
        """
        self.xp = get_array_module()
        self.pre_shape = pre_shape
        self.post_shape = post_shape
        self.dt = dt

        # Convert delays and weights to device
        self.delays = to_device(delays.astype(np.float32))
        self.weights = to_device(weights.astype(np.float32))

        # Calculate maximum delay in timesteps
        max_delay = float(to_numpy(self.xp.max(delays)))
        self.max_delay_steps = int(np.ceil(max_delay / dt)) + 1

        # Initialize delay buffer for presynaptic spikes
        pre_size = int(np.prod(pre_shape))
        self.delay_buffer = self.xp.zeros(
            (self.max_delay_steps, pre_size),
            dtype=self.xp.float32
        )
        self.buffer_index = 0

    def reset(self):
        """Reset delay buffer."""
        self.delay_buffer.fill(0.0)
        self.buffer_index = 0

    def propagate(self, pre_spikes):
        """
        Propagate spikes through the synapse with delays.

        Args:
            pre_spikes: Presynaptic spike array

        Returns:
            Postsynaptic input current
        """
        xp = self.xp

        # Flatten presynaptic spikes
        pre_flat = pre_spikes.flatten()

        # Add current spikes to delay buffer
        self.delay_buffer[self.buffer_index] = pre_flat

        # Advance buffer index
        self.buffer_index = (self.buffer_index + 1) % self.max_delay_steps

        # This is a simplified version - full implementation would use
        # delay-specific retrieval. For now, we use a weighted sum approach.
        return self._compute_delayed_input()

    def _compute_delayed_input(self):
        """Compute delayed input to postsynaptic neurons."""
        # This is a placeholder - subclasses implement specific connectivity patterns
        raise NotImplementedError


class SpikeMonitor:
    """
    Monitor to record spike activity from a neuron group.
    """

    def __init__(self, neuron_group, max_spikes=100000):
        """
        Initialize spike monitor.

        Args:
            neuron_group: Neuron group to monitor
            max_spikes: Maximum number of spikes to store
        """
        self.neuron_group = neuron_group
        self.max_spikes = max_spikes
        self.xp = get_array_module()

        # Storage for spike times and neuron indices
        self.times = []
        self.indices = []

    def record(self):
        """Record current spikes."""
        spikes = to_numpy(self.neuron_group.spikes)
        spike_idx = np.where(spikes.flatten() > 0)[0]

        if len(spike_idx) > 0:
            current_time = self.neuron_group.current_time
            self.times.extend([current_time] * len(spike_idx))
            self.indices.extend(spike_idx.tolist())

    def reset(self):
        """Clear recorded data."""
        self.times = []
        self.indices = []

    def get_data(self):
        """Get recorded spike data."""
        return {
            'times': np.array(self.times),
            'indices': np.array(self.indices),
            'shape': self.neuron_group.shape
        }

    def get_firing_rates(self, time_window=None):
        """
        Calculate firing rates for each neuron.

        Args:
            time_window: Tuple of (start, end) times. If None, uses all recorded data.

        Returns:
            Array of firing rates (Hz)
        """
        times = np.array(self.times)
        indices = np.array(self.indices)

        if len(times) == 0:
            return np.zeros(self.neuron_group.n_neurons)

        if time_window is not None:
            mask = (times >= time_window[0]) & (times <= time_window[1])
            times = times[mask]
            indices = indices[mask]
            duration = time_window[1] - time_window[0]
        else:
            duration = times.max() - times.min() if len(times) > 1 else 1.0

        # Count spikes per neuron
        spike_counts = np.bincount(indices, minlength=self.neuron_group.n_neurons)

        # Convert to Hz
        firing_rates = spike_counts / (duration / 1000.0)  # duration is in ms

        return firing_rates.reshape(self.neuron_group.shape)


if __name__ == "__main__":
    # Test neuron models
    import matplotlib.pyplot as plt

    print("Testing LIF neurons...")

    # Create a small group of neurons
    neurons = LIFNeuronGroup((10, 10))
    monitor = SpikeMonitor(neurons)

    # Simulate with varying input
    n_steps = 1000
    v_history = []

    for i in range(n_steps):
        # Sinusoidal input current
        t = i * DT
        current = 20 * np.sin(2 * np.pi * t / 200) + 10  # Oscillating input
        current_array = np.full(neurons.shape, current, dtype=np.float32)

        neurons.step(current_array)
        monitor.record()

        # Record voltage of one neuron
        v_history.append(to_numpy(neurons.v[5, 5]))

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Membrane potential
    axes[0].plot(np.arange(n_steps) * DT, v_history)
    axes[0].axhline(neurons.v_threshold, color='r', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Membrane potential (mV)')
    axes[0].set_title('Single neuron membrane potential')
    axes[0].legend()

    # Raster plot
    spike_data = monitor.get_data()
    if len(spike_data['times']) > 0:
        axes[1].scatter(spike_data['times'], spike_data['indices'], s=1, alpha=0.5)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Neuron index')
    axes[1].set_title('Population raster plot')

    plt.tight_layout()
    plt.savefig('neuron_test.png', dpi=100)
    plt.close()

    print(f"Simulated {n_steps} timesteps")
    print(f"Total spikes: {len(spike_data['times'])}")
    print("Saved test figure to neuron_test.png")
