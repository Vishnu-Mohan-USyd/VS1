# Visual Cortex Orientation Selectivity Model

A spiking neural network implementing the emergence of orientation selectivity in V1 through the visual pathway: **RGC (ON/OFF) -> LGN -> V1 Layer 4**.

## Overview

This model implements the hypothesis that orientation selectivity emerges from:

1. **Retinotopic LGN patches** projecting to multiple V1 orientation ensembles
2. **STDP-based learning** that strengthens orientation-specific connections
3. **Lateral competition** ensuring diverse orientation preferences within each hypercolumn

### Architecture

```
Visual Input (Drifting Grating)
        |
        v
+-------------------+
|  RGC Layer        |
|  ON/OFF channels  |
+-------------------+
        |
        v
+-------------------+
|  LGN Layer        |
|  ON/OFF channels  |
|  (Thalamic relay) |
+-------------------+
        |
        v
+-------------------+
|  V1 Layer 4       |
|  Excitatory (RS)  |
|  Inhibitory (FS)  |
+-------------------+
```

### Key Features

- **Izhikevich neurons** with biologically realistic spike patterns
  - Regular Spiking (RS) for V1 excitatory neurons
  - Fast Spiking (FS) for V1 inhibitory interneurons
  - Thalamic (TC) for LGN relay neurons

- **ON/OFF visual pathways** with Gabor-like receptive field initialization

- **STDP with soft bounds** for stable learning

- **Strong lateral inhibition** for winner-take-all competition within orientation columns

- **GPU acceleration** via CuPy (optional, falls back to NumPy)

## Installation

### Requirements

```bash
pip install numpy matplotlib
```

For GPU acceleration (optional):
```bash
pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version
```

## Usage

### Quick Start

```python
from visual_cortex_model import VisualCortexModel

# Create model
model = VisualCortexModel(
    visual_field=(16, 16),    # Input size
    patch_size=4,              # LGN patch size
    n_orientations=8,          # Orientations per hypercolumn
    gabor_strength=0.9,        # Initial orientation bias
    stdp_rate=0.015            # Learning rate
)

# Train on drifting gratings
history = model.train(
    n_epochs=25,
    orientations=[0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5],
    trials_per_ori=12,
    duration=400,
    visualize_every=5,
    save_dir="output"
)
```

### Run from Command Line

```bash
python visual_cortex_model.py
```

This will:
1. Create a 16x16 visual field model
2. Train for 25 epochs on 8 orientations
3. Generate visualizations in the `output/` directory

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `visual_field` | (width, height) of input | (16, 16) |
| `patch_size` | Size of each LGN patch | 4 |
| `n_orientations` | Number of orientation preferences | 8 |
| `gabor_strength` | Initial Gabor weight bias (0-1) | 0.9 |
| `stdp_rate` | STDP learning rate | 0.015 |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_epochs` | Number of training epochs | 25 |
| `orientations` | List of orientations (degrees) | [0, 22.5, ..., 157.5] |
| `trials_per_ori` | Trials per orientation per epoch | 12 |
| `duration` | Trial duration (ms) | 400 |
| `visualize_every` | Visualization frequency | 5 |

## Output Visualizations

The training process generates several visualization files:

1. **Receptive Fields** (`rf_epoch_XXX.png`)
   - ON-OFF weight patterns showing the effective receptive field
   - Each column shows one orientation preference

2. **Tuning Curves** (`tuning_epoch_XXX.png`)
   - Response vs stimulus orientation for each neuron
   - Comparison of learned vs expected preferred orientations

3. **OSI Distribution** (`osi_epoch_XXX.png`)
   - Histogram of Orientation Selectivity Index
   - OSI = (R_pref - R_orth) / (R_pref + R_orth)

4. **Orientation Map** (`ori_map_epoch_XXX.png`)
   - Spatial arrangement of preferred orientations
   - Color indicates orientation, saturation indicates selectivity

5. **Training History** (`training_history.png`)
   - Mean OSI over training epochs

## Model Details

### Neuron Model

Uses the Izhikevich model:
```
dv/dt = 0.04*v^2 + 5*v + 140 - u + I
du/dt = a*(b*v - u)
if v >= 30: v = c, u = u + d
```

### STDP Rule

Soft-bounds STDP:
- Potentiation: `dw = A+ * (w_max - w) * pre_trace * post_spike`
- Depression: `dw = -A- * (w - w_min) * pre_spike * post_trace`

### Connectivity

- **LGN -> V1**: Each 4x4 LGN patch projects to all 8 V1 neurons in its hypercolumn
- **V1 Lateral**: Within-hypercolumn inhibition via interneurons

## Example Results

After training, typical results show:
- Initial Mean OSI: ~0.03
- Final Mean OSI: ~0.05-0.08
- Receptive fields develop oriented structure
- Neurons show preference for specific orientations

## File Structure

```
VS1/
├── visual_cortex_model.py     # Main model (recommended)
├── orientation_selectivity_network.py    # Alternative v1
├── orientation_selectivity_network_v2.py # Alternative v2
├── orientation_selectivity_network_v3.py # Alternative v3
├── requirements.txt
├── README.md
└── output/                    # Generated visualizations
```

## References

- Izhikevich (2003) "Simple Model of Spiking Neurons"
- Hubel & Wiesel - Simple cell receptive fields
- Srinivasa & Jiang (2013) - Development model with continuous plasticity

## License

MIT License
