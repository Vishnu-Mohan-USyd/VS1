"""
Configuration parameters for the V1 Orientation Selectivity Network.

This module contains all hyperparameters organized by component.
Parameters are based on biological data from the literature.
"""

import numpy as np

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
SIM_DT = 1.0  # Time step in ms (1ms for faster simulation)
SIM_DURATION_TRAIN = 5000  # Training duration per orientation in ms
SIM_DURATION_TEST = 1000  # Test duration per orientation in ms

# =============================================================================
# NETWORK DIMENSIONS
# =============================================================================
# Retinal/LGN dimensions (visual field representation)
RETINA_SIZE = 32  # 32x32 grid for RGC/LGN (smaller for faster simulation)

# LGN patch size that projects to one hypercolumn
LGN_PATCH_SIZE = 8  # 8x8 patch

# Number of hypercolumns (derived from retina size and patch size with overlap)
HYPERCOLUMN_STRIDE = 8  # Stride between hypercolumn centers (non-overlapping)
N_HYPERCOLUMNS_X = (RETINA_SIZE - LGN_PATCH_SIZE) // HYPERCOLUMN_STRIDE + 1
N_HYPERCOLUMNS_Y = (RETINA_SIZE - LGN_PATCH_SIZE) // HYPERCOLUMN_STRIDE + 1
N_HYPERCOLUMNS = N_HYPERCOLUMNS_X * N_HYPERCOLUMNS_Y

# Orientation columns per hypercolumn
N_ORIENTATIONS = 8  # 8 orientation preferences (0, 22.5, 45, ..., 157.5 degrees)

# Neurons per ensemble (orientation column)
NEURONS_PER_ENSEMBLE = 4  # 4 excitatory neurons per orientation ensemble (smaller)
INHIBITORY_PER_HYPERCOLUMN = 8  # Inhibitory neurons per hypercolumn

# =============================================================================
# IZHIKEVICH NEURON PARAMETERS
# Based on Izhikevich (2003) "Simple Model of Spiking Neurons"
# and Izhikevich (2004) "Which Model to Use for Cortical Spiking Neurons?"
# =============================================================================

# Regular Spiking (RS) - V1 L4 Excitatory neurons
# These show spike frequency adaptation
IZH_RS = {
    'a': 0.02,   # Recovery time constant (smaller = slower recovery)
    'b': 0.2,    # Sensitivity of recovery to subthreshold fluctuations
    'c': -65.0,  # After-spike reset value of membrane potential (mV)
    'd': 8.0,    # After-spike reset of recovery variable
    'v_thresh': 30.0,  # Spike threshold (mV)
    'v_rest': -65.0,   # Resting potential (mV)
}

# Fast Spiking (FS) - V1 L4 Inhibitory neurons (PV+ basket cells)
# No spike frequency adaptation, high firing rates
IZH_FS = {
    'a': 0.1,    # Faster recovery
    'b': 0.2,
    'c': -65.0,
    'd': 2.0,    # Smaller d = less adaptation
    'v_thresh': 30.0,
    'v_rest': -65.0,
}

# Tonic Spiking - RGC neurons
# Sustained response to constant input
IZH_TONIC = {
    'a': 0.02,
    'b': 0.2,
    'c': -65.0,
    'd': 6.0,
    'v_thresh': 30.0,
    'v_rest': -65.0,
}

# Thalamocortical (TC) - LGN relay neurons
# Based on Izhikevich (2003) thalamocortical parameters
IZH_TC = {
    'a': 0.02,
    'b': 0.25,
    'c': -65.0,
    'd': 0.05,
    'v_thresh': 30.0,
    'v_rest': -65.0,
}

# =============================================================================
# SYNAPTIC PARAMETERS
# =============================================================================

# Synaptic time constants (ms)
TAU_AMPA = 5.0    # AMPA (fast excitatory)
TAU_NMDA = 100.0  # NMDA (slow excitatory, for STDP)
TAU_GABA_A = 10.0  # GABA_A (fast inhibitory)

# Reversal potentials (mV)
E_AMPA = 0.0
E_NMDA = 0.0
E_GABA = -75.0

# =============================================================================
# CONNECTION PARAMETERS
# =============================================================================

# RGC -> LGN (one-to-one, strong)
W_RGC_LGN = 3.0  # Relay connection

# LGN -> V1 (plastic, initially very weak)
W_LGN_V1_INIT_MEAN = 0.005  # Very weak initial (need many spikes to drive V1)
W_LGN_V1_INIT_STD = 0.002   # Initial weight std
W_LGN_V1_MIN = 0.0          # Minimum weight (for STDP bounds)
W_LGN_V1_MAX = 0.15         # Maximum weight (for STDP bounds)

# Conduction delays (ms)
DELAY_RGC_LGN = 2.0       # Fast relay
DELAY_LGN_V1_MIN = 1.0    # Minimum thalamocortical delay
DELAY_LGN_V1_MAX = 8.0    # Maximum thalamocortical delay (varied per projection)

# V1 Lateral connections
W_V1_EXC_LOCAL = 0.05     # Excitatory weight (for E->I)
W_V1_INH = 0.8            # Strong inhibitory strength for winner-take-all

# Connection probabilities
P_LGN_V1 = 0.2            # Probability of LGN->V1 connection (sparse for diversity)
P_V1_LATERAL_EXC = 0.1    # Probability of lateral excitatory connection
P_V1_LATERAL_INH = 0.5    # Higher probability for inhibition

# =============================================================================
# STDP PARAMETERS
# Based on Bi & Poo (1998) and Song et al. (2000)
# =============================================================================

STDP_TAU_PLUS = 20.0   # Time constant for potentiation (ms)
STDP_TAU_MINUS = 20.0  # Time constant for depression (ms)
STDP_A_PLUS = 0.003    # Maximum potentiation (small for stable learning)
STDP_A_MINUS = 0.0035  # Maximum depression (slightly larger for stability)
STDP_W_MAX = W_LGN_V1_MAX  # Maximum weight

# =============================================================================
# STIMULUS PARAMETERS
# =============================================================================

# Drifting grating parameters
GRATING_SPATIAL_FREQ = 0.15  # Cycles per pixel
GRATING_TEMPORAL_FREQ = 2.0  # Hz (drift speed)
GRATING_CONTRAST = 1.0       # Full contrast
GRATING_ORIENTATIONS = np.linspace(0, 180, N_ORIENTATIONS, endpoint=False)  # Degrees

# Spike encoding
SPIKE_RATE_MAX = 100.0  # Maximum firing rate for ON/OFF cells (Hz)
SPIKE_RATE_BASELINE = 5.0  # Baseline firing rate (Hz)

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

N_TRAINING_EPOCHS = 3     # Number of full training cycles through all orientations
N_PRESENTATIONS_PER_ORI = 10  # Presentations per orientation per epoch
PRESENTATION_DURATION = 500  # Duration of each grating presentation (ms)
INTER_STIMULUS_INTERVAL = 100  # Gap between presentations (ms)

# =============================================================================
# VISUALIZATION
# =============================================================================

PLOT_INTERVAL = 1000  # Plot every N ms during training
SAVE_WEIGHTS_INTERVAL = 5000  # Save weights every N ms
