"""
Configuration parameters for the spiking neural network model.
RGC -> LGN -> V1 (L4) orientation selectivity model.
"""

import numpy as np

# =============================================================================
# Simulation Parameters
# =============================================================================
DT = 1.0  # Time step in ms
SIMULATION_TIME = 500  # Total simulation time per stimulus in ms

# =============================================================================
# Visual Field / Stimulus Parameters
# =============================================================================
VISUAL_FIELD_SIZE = 32  # Size of visual field (pixels)
GRATING_SPATIAL_FREQ = 0.1  # Spatial frequency (cycles per pixel)
GRATING_TEMPORAL_FREQ = 2.0  # Temporal frequency (Hz)
GRATING_CONTRAST = 1.0  # Contrast [0, 1]
N_ORIENTATIONS = 8  # Number of orientations to train on
ORIENTATIONS = np.linspace(0, 180, N_ORIENTATIONS, endpoint=False)  # degrees

# =============================================================================
# RGC Layer Parameters
# =============================================================================
RGC_SIZE = VISUAL_FIELD_SIZE  # RGC array size (ON and OFF each)
RGC_CENTER_SIGMA = 1.0  # Center Gaussian sigma
RGC_SURROUND_SIGMA = 3.0  # Surround Gaussian sigma
RGC_CENTER_WEIGHT = 1.0  # Center weight
RGC_SURROUND_WEIGHT = 0.5  # Surround weight
RGC_THRESHOLD = 0.3  # Firing threshold
RGC_SPONTANEOUS_RATE = 0.001  # Spontaneous firing probability per timestep

# =============================================================================
# LGN Layer Parameters
# =============================================================================
LGN_SIZE = RGC_SIZE  # LGN maintains retinotopic mapping
LGN_RGC_WEIGHT = 1.0  # Weight from RGC to LGN

# LIF neuron parameters for LGN
LGN_V_REST = -70.0  # mV
LGN_V_RESET = -75.0  # mV
LGN_V_THRESHOLD = -55.0  # mV, firing threshold
LGN_TAU_M = 10.0  # ms, membrane time constant
LGN_REFRACTORY = 2.0  # ms

# =============================================================================
# V1 Layer Parameters
# =============================================================================
LGN_PATCH_SIZE = 4  # Size of LGN patch projecting to V1 ensembles (4x4)
N_V1_ENSEMBLES = 8  # Number of V1 ensembles per hypercolumn (orientation column)
V1_ENSEMBLE_SIZE = 4  # Number of neurons per ensemble

# Calculate number of hypercolumns based on LGN size and patch size
N_HYPERCOLUMNS_X = LGN_SIZE // LGN_PATCH_SIZE
N_HYPERCOLUMNS_Y = LGN_SIZE // LGN_PATCH_SIZE
N_HYPERCOLUMNS = N_HYPERCOLUMNS_X * N_HYPERCOLUMNS_Y

# LIF parameters for V1 neurons
V1_V_REST = -70.0  # mV
V1_V_RESET = -75.0  # mV
V1_V_THRESHOLD = -55.0  # mV
V1_TAU_M = 20.0  # ms, membrane time constant
V1_REFRACTORY = 3.0  # ms

# =============================================================================
# Connectivity Parameters
# =============================================================================
# Initial weight range for LGN -> V1 projections
W_LGN_V1_INIT_MIN = 0.3
W_LGN_V1_INIT_MAX = 0.7

# Initial conduction delay range (ms)
DELAY_MIN = 1.0
DELAY_MAX = 10.0

# =============================================================================
# STDP Parameters
# =============================================================================
STDP_TAU_PLUS = 20.0  # ms, time constant for potentiation
STDP_TAU_MINUS = 25.0  # ms, time constant for depression (wider window)
STDP_A_PLUS = 0.015  # Learning rate for potentiation (stronger LTP)
STDP_A_MINUS = 0.008  # Learning rate for depression (weaker LTD to preserve selectivity)
STDP_W_MAX = 1.0  # Maximum weight
STDP_W_MIN = 0.0  # Minimum weight

# =============================================================================
# Lateral Inhibition Parameters
# =============================================================================
LATERAL_INHIBITION_STRENGTH = 0.5  # Strength of lateral inhibition between ensembles
LATERAL_INHIBITION_TAU = 5.0  # ms, time constant for inhibition decay

# =============================================================================
# Training Parameters
# =============================================================================
N_TRAINING_EPOCHS = 10  # Number of training epochs
N_PRESENTATIONS_PER_ORIENTATION = 5  # Presentations per orientation per epoch
VISUALIZATION_INTERVAL = 1  # Visualize every N epochs

# =============================================================================
# GPU Configuration
# =============================================================================
USE_GPU = True  # Set to False to use NumPy instead of CuPy
