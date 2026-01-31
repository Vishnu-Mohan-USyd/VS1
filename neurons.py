"""
Izhikevich neuron implementation with GPU acceleration via CuPy.

References:
- Izhikevich (2003) "Simple Model of Spiking Neurons" IEEE Trans Neural Netw
- Izhikevich (2004) "Which Model to Use for Cortical Spiking Neurons?" IEEE Trans Neural Netw
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
    print("CuPy available - using GPU acceleration")
except ImportError:
    import numpy as np
    GPU_AVAILABLE = False
    xp = np
    print("CuPy not available - using CPU (NumPy)")

import numpy as np
from config import (
    IZH_RS, IZH_FS, IZH_TONIC, IZH_TC,
    TAU_AMPA, TAU_GABA_A, E_AMPA, E_GABA, SIM_DT
)


def to_numpy(arr):
    """Convert array to numpy (handles both cupy and numpy)."""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return arr


def to_gpu(arr):
    """Convert array to GPU if available."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


class IzhikevichPopulation:
    """
    Population of Izhikevich neurons with conductance-based synapses.

    The Izhikevich model equations:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        if v >= 30 mV: v = c, u = u + d

    With conductance-based synapses:
        I = g_exc * (E_exc - v) + g_inh * (E_inh - v) + I_ext
    """

    def __init__(self, n_neurons, neuron_type='RS', name='population'):
        """
        Initialize a population of Izhikevich neurons.

        Args:
            n_neurons: Number of neurons
            neuron_type: One of 'RS', 'FS', 'TONIC', 'TC'
            name: Population name for debugging
        """
        self.n_neurons = n_neurons
        self.name = name
        self.neuron_type = neuron_type

        # Get parameters based on neuron type
        params = {
            'RS': IZH_RS,
            'FS': IZH_FS,
            'TONIC': IZH_TONIC,
            'TC': IZH_TC
        }[neuron_type]

        # Store parameters as arrays for vectorized computation
        self.a = xp.full(n_neurons, params['a'], dtype=xp.float32)
        self.b = xp.full(n_neurons, params['b'], dtype=xp.float32)
        self.c = xp.full(n_neurons, params['c'], dtype=xp.float32)
        self.d = xp.full(n_neurons, params['d'], dtype=xp.float32)
        self.v_thresh = params['v_thresh']
        self.v_rest = params['v_rest']

        # Add some heterogeneity (biological variability)
        noise = 0.05
        self.a *= (1 + noise * xp.random.randn(n_neurons).astype(xp.float32))
        self.b *= (1 + noise * xp.random.randn(n_neurons).astype(xp.float32))
        self.c += noise * 5 * xp.random.randn(n_neurons).astype(xp.float32)
        self.d *= (1 + noise * xp.random.randn(n_neurons).astype(xp.float32))

        # State variables
        self.v = xp.full(n_neurons, self.v_rest, dtype=xp.float32)  # Membrane potential
        self.u = self.b * self.v  # Recovery variable

        # Conductances
        self.g_exc = xp.zeros(n_neurons, dtype=xp.float32)  # Excitatory conductance
        self.g_inh = xp.zeros(n_neurons, dtype=xp.float32)  # Inhibitory conductance

        # External current (for direct stimulation)
        self.I_ext = xp.zeros(n_neurons, dtype=xp.float32)

        # Spike tracking
        self.spikes = xp.zeros(n_neurons, dtype=xp.bool_)
        self.spike_times = [[] for _ in range(n_neurons)]  # For STDP
        self.last_spike_time = xp.full(n_neurons, -1000.0, dtype=xp.float32)

        # Synaptic time constants
        self.tau_exc = TAU_AMPA
        self.tau_inh = TAU_GABA_A

    def reset(self):
        """Reset all neurons to resting state."""
        self.v = xp.full(self.n_neurons, self.v_rest, dtype=xp.float32)
        self.u = self.b * self.v
        self.g_exc = xp.zeros(self.n_neurons, dtype=xp.float32)
        self.g_inh = xp.zeros(self.n_neurons, dtype=xp.float32)
        self.I_ext = xp.zeros(self.n_neurons, dtype=xp.float32)
        self.spikes = xp.zeros(self.n_neurons, dtype=xp.bool_)
        self.spike_times = [[] for _ in range(self.n_neurons)]
        self.last_spike_time = xp.full(self.n_neurons, -1000.0, dtype=xp.float32)

    def step(self, t, dt=SIM_DT):
        """
        Advance the simulation by one time step.

        Args:
            t: Current simulation time (ms)
            dt: Time step (ms)

        Returns:
            Boolean array of spikes this timestep
        """
        # Compute total synaptic current (conductance-based)
        I_syn = self.g_exc * (E_AMPA - self.v) + self.g_inh * (E_GABA - self.v)
        I_total = I_syn + self.I_ext

        # Izhikevich dynamics (using smaller substeps for numerical stability)
        n_substeps = 2
        sub_dt = dt / n_substeps

        for _ in range(n_substeps):
            # dv/dt = 0.04*v^2 + 5*v + 140 - u + I
            dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I_total) * sub_dt
            # du/dt = a*(b*v - u)
            du = self.a * (self.b * self.v - self.u) * sub_dt

            self.v += dv
            self.u += du

            # Clamp voltage to prevent numerical explosion
            self.v = xp.clip(self.v, -100, self.v_thresh + 10)

        # Detect spikes
        self.spikes = self.v >= self.v_thresh

        # Reset spiking neurons
        spike_indices = xp.where(self.spikes)[0]
        if len(spike_indices) > 0:
            self.v[spike_indices] = self.c[spike_indices]
            self.u[spike_indices] = self.u[spike_indices] + self.d[spike_indices]
            self.last_spike_time[spike_indices] = t

            # Record spike times (convert to numpy for list operations)
            spike_idx_np = to_numpy(spike_indices)
            for idx in spike_idx_np:
                self.spike_times[idx].append(t)

        # Decay conductances
        self.g_exc *= xp.exp(-dt / self.tau_exc)
        self.g_inh *= xp.exp(-dt / self.tau_inh)

        return self.spikes

    def receive_spikes(self, weights, presynaptic_spikes, excitatory=True):
        """
        Receive spikes from presynaptic neurons.

        Args:
            weights: Weight matrix (pre x post) or sparse representation
            presynaptic_spikes: Boolean array of presynaptic spikes
            excitatory: Whether these are excitatory synapses
        """
        if not xp.any(presynaptic_spikes):
            return

        # Compute postsynaptic conductance change
        # weights[i,j] = weight from pre neuron i to post neuron j
        if excitatory:
            self.g_exc += xp.dot(presynaptic_spikes.astype(xp.float32), weights)
        else:
            self.g_inh += xp.dot(presynaptic_spikes.astype(xp.float32), weights)

    def get_firing_rates(self, time_window=100.0):
        """
        Compute firing rates based on recent spike times.

        Args:
            time_window: Time window to count spikes (ms)

        Returns:
            Array of firing rates (Hz)
        """
        rates = xp.zeros(self.n_neurons, dtype=xp.float32)
        current_time = float(to_numpy(self.last_spike_time.max())) + 1

        for i in range(self.n_neurons):
            recent_spikes = sum(1 for t in self.spike_times[i]
                              if t > current_time - time_window)
            rates[i] = recent_spikes * 1000.0 / time_window  # Convert to Hz

        return rates

    def clear_spike_history(self, keep_last_n=100):
        """Clear old spike times to save memory."""
        for i in range(self.n_neurons):
            if len(self.spike_times[i]) > keep_last_n:
                self.spike_times[i] = self.spike_times[i][-keep_last_n:]


class SpikeDelayBuffer:
    """
    Buffer to handle axonal conduction delays.

    Stores spikes and delivers them after the appropriate delay.
    """

    def __init__(self, n_neurons, max_delay, dt=SIM_DT):
        """
        Initialize delay buffer.

        Args:
            n_neurons: Number of source neurons
            max_delay: Maximum delay in ms
            dt: Simulation time step
        """
        self.n_neurons = n_neurons
        self.max_delay = max_delay
        self.dt = dt
        self.buffer_size = int(np.ceil(max_delay / dt)) + 1

        # Circular buffer: buffer[t % buffer_size] contains spikes from time t
        self.buffer = xp.zeros((self.buffer_size, n_neurons), dtype=xp.bool_)
        self.current_idx = 0

    def add_spikes(self, spikes):
        """Add current spikes to the buffer."""
        self.buffer[self.current_idx] = spikes

    def get_delayed_spikes(self, delays):
        """
        Get spikes after specified delays.

        Args:
            delays: Array of delays in ms for each neuron
                   (if all same value, uses fast path)

        Returns:
            Array of spikes that arrive now given their delays
        """
        # Fast path: if all delays are the same (common case)
        if xp.all(delays == delays[0]):
            delay_steps = int(delays[0] / self.dt)
            delay_steps = min(max(delay_steps, 0), self.buffer_size - 1)
            past_idx = (self.current_idx - delay_steps) % self.buffer_size
            return self.buffer[past_idx].copy()

        # Slow path: different delays per neuron
        delay_steps = (delays / self.dt).astype(xp.int32)
        delay_steps = xp.clip(delay_steps, 0, self.buffer_size - 1)

        # Vectorized approach
        unique_delays = xp.unique(delay_steps)
        output = xp.zeros(self.n_neurons, dtype=xp.bool_)
        for d in to_numpy(unique_delays):
            mask = delay_steps == d
            past_idx = (self.current_idx - d) % self.buffer_size
            output = output | (self.buffer[past_idx] & mask)

        return output

    def step(self):
        """Advance buffer to next time step."""
        self.current_idx = (self.current_idx + 1) % self.buffer_size
        self.buffer[self.current_idx] = False

    def reset(self):
        """Clear the buffer."""
        self.buffer = xp.zeros((self.buffer_size, self.n_neurons), dtype=xp.bool_)
        self.current_idx = 0
