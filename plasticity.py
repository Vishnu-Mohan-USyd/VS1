"""
Spike-Timing Dependent Plasticity (STDP) implementation.

References:
- Bi & Poo (1998) "Synaptic Modifications in Cultured Hippocampal Neurons"
- Song, Miller & Abbott (2000) "Competitive Hebbian learning through STDP"
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
except ImportError:
    import numpy as np
    GPU_AVAILABLE = False
    xp = np

import numpy as np
from config import (
    STDP_TAU_PLUS, STDP_TAU_MINUS, STDP_A_PLUS, STDP_A_MINUS,
    W_LGN_V1_MIN, W_LGN_V1_MAX, SIM_DT
)
from neurons import to_numpy, to_gpu


class STDP:
    """
    Pair-based STDP with eligibility traces.

    The learning rule:
        - Pre before post (causal): Potentiation, dw = A+ * exp(-dt/tau+)
        - Post before pre (acausal): Depression, dw = -A- * exp(-dt/tau-)

    Uses eligibility traces for efficient implementation:
        - x_pre: trace of presynaptic activity (increases on pre spike)
        - x_post: trace of postsynaptic activity (increases on post spike)

    Weight update:
        - On pre spike: w += A- * x_post (depression from post-before-pre)
        - On post spike: w += A+ * x_pre (potentiation from pre-before-post)
    """

    def __init__(self, n_pre, n_post, weights, w_min=W_LGN_V1_MIN, w_max=W_LGN_V1_MAX):
        """
        Initialize STDP for a synaptic projection.

        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            weights: Initial weight matrix (n_pre x n_post)
            w_min: Minimum weight
            w_max: Maximum weight
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.w_min = w_min
        self.w_max = w_max

        # Weights (n_pre x n_post)
        self.weights = to_gpu(weights.astype(np.float32))

        # Connection mask (which synapses exist)
        self.mask = self.weights > 0

        # STDP parameters
        self.tau_plus = STDP_TAU_PLUS
        self.tau_minus = STDP_TAU_MINUS
        self.A_plus = STDP_A_PLUS
        self.A_minus = STDP_A_MINUS

        # Eligibility traces
        self.x_pre = xp.zeros(n_pre, dtype=xp.float32)   # Pre-synaptic trace
        self.x_post = xp.zeros(n_post, dtype=xp.float32)  # Post-synaptic trace

        # For tracking weight changes
        self.total_potentiation = 0.0
        self.total_depression = 0.0

    def update(self, pre_spikes, post_spikes, dt=SIM_DT, learning_rate=1.0):
        """
        Update weights based on pre and post spikes.

        Args:
            pre_spikes: Boolean array of presynaptic spikes
            post_spikes: Boolean array of postsynaptic spikes
            dt: Time step (ms)
            learning_rate: Scaling factor for weight updates
        """
        # Convert to float for computation
        pre_spikes_f = pre_spikes.astype(xp.float32)
        post_spikes_f = post_spikes.astype(xp.float32)

        # Weight updates from STDP
        # When post spikes: potentiate synapses from recently active pre neurons
        if xp.any(post_spikes):
            # dw = A+ * x_pre for all pre->post connections where post spiked
            # Shape: (n_pre,) x (n_post,) -> update weights[:, post_spiked]
            dw_pot = learning_rate * self.A_plus * xp.outer(self.x_pre, post_spikes_f)
            dw_pot *= self.mask  # Only update existing connections
            self.weights += dw_pot
            self.total_potentiation += float(to_numpy(xp.sum(dw_pot)))

        # When pre spikes: depress synapses to recently active post neurons
        if xp.any(pre_spikes):
            # dw = -A- * x_post for all pre->post connections where pre spiked
            dw_dep = -learning_rate * self.A_minus * xp.outer(pre_spikes_f, self.x_post)
            dw_dep *= self.mask
            self.weights += dw_dep
            self.total_depression += float(to_numpy(xp.sum(xp.abs(dw_dep))))

        # Update eligibility traces
        # Decay traces
        self.x_pre *= xp.exp(-dt / self.tau_plus)
        self.x_post *= xp.exp(-dt / self.tau_minus)

        # Increment traces on spikes
        self.x_pre += pre_spikes_f
        self.x_post += post_spikes_f

        # Enforce weight bounds
        self.weights = xp.clip(self.weights, self.w_min, self.w_max)

        # Keep mask consistent (don't create new connections)
        self.weights *= self.mask

    def get_weights(self):
        """Return weights as numpy array."""
        return to_numpy(self.weights)

    def reset_traces(self):
        """Reset eligibility traces (e.g., between trials)."""
        self.x_pre = xp.zeros(self.n_pre, dtype=xp.float32)
        self.x_post = xp.zeros(self.n_post, dtype=xp.float32)

    def get_stats(self):
        """Get weight statistics."""
        w = to_numpy(self.weights)
        w_active = w[to_numpy(self.mask)]
        return {
            'mean': np.mean(w_active) if len(w_active) > 0 else 0,
            'std': np.std(w_active) if len(w_active) > 0 else 0,
            'min': np.min(w_active) if len(w_active) > 0 else 0,
            'max': np.max(w_active) if len(w_active) > 0 else 0,
            'total_pot': self.total_potentiation,
            'total_dep': self.total_depression,
        }


class InhibitorySTDP:
    """
    Inhibitory STDP for maintaining E/I balance.

    Based on Vogels et al. (2011) - inhibitory plasticity that maintains
    a target postsynaptic firing rate.

    Rule: Potentiate inhibition when post fires (to reduce future firing)
          Depress inhibition when pre fires alone (to allow firing)
    """

    def __init__(self, n_pre, n_post, weights, target_rate=10.0):
        """
        Initialize inhibitory STDP.

        Args:
            n_pre: Number of presynaptic (inhibitory) neurons
            n_post: Number of postsynaptic neurons
            weights: Initial weight matrix (n_pre x n_post)
            target_rate: Target firing rate for postsynaptic neurons (Hz)
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.target_rate = target_rate

        self.weights = to_gpu(weights.astype(np.float32))
        self.mask = self.weights > 0

        # Learning parameters
        self.tau = 20.0  # Trace time constant
        self.eta = 0.001  # Learning rate
        self.alpha = 2 * self.target_rate * self.tau / 1000  # Balance factor

        # Traces
        self.x_pre = xp.zeros(n_pre, dtype=xp.float32)
        self.x_post = xp.zeros(n_post, dtype=xp.float32)

        self.w_min = 0.0
        self.w_max = 20.0

    def update(self, pre_spikes, post_spikes, dt=SIM_DT, learning_rate=1.0):
        """Update inhibitory weights."""
        pre_spikes_f = pre_spikes.astype(xp.float32)
        post_spikes_f = post_spikes.astype(xp.float32)

        # On post spike: strengthen inhibition from recently active pre
        if xp.any(post_spikes):
            dw = learning_rate * self.eta * xp.outer(self.x_pre, post_spikes_f)
            dw *= self.mask
            self.weights += dw

        # On pre spike: weaken inhibition to recently active post (minus baseline)
        if xp.any(pre_spikes):
            dw = learning_rate * self.eta * xp.outer(pre_spikes_f, self.x_post - self.alpha)
            dw *= self.mask
            self.weights += dw

        # Update traces
        self.x_pre *= xp.exp(-dt / self.tau)
        self.x_post *= xp.exp(-dt / self.tau)
        self.x_pre += pre_spikes_f
        self.x_post += post_spikes_f

        # Bounds
        self.weights = xp.clip(self.weights, self.w_min, self.w_max)
        self.weights *= self.mask

    def get_weights(self):
        """Return weights as numpy array."""
        return to_numpy(self.weights)

    def reset_traces(self):
        """Reset eligibility traces."""
        self.x_pre = xp.zeros(self.n_pre, dtype=xp.float32)
        self.x_post = xp.zeros(self.n_post, dtype=xp.float32)
