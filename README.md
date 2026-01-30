# Spiking Neural Network for Orientation Selectivity

A biologically-inspired spiking neural network that simulates the early visual pathway (RGC -> LGN -> V1) and demonstrates the emergence of orientation selectivity through STDP learning.

## Architecture

```
                    Visual Stimulus (Drifting Gratings)
                                |
                                v
    +----------------------------------------------------------+
    |                    RGC Layer                              |
    |   ON cells (center-bright)  |  OFF cells (center-dark)   |
    +----------------------------------------------------------+
                                |
                                v
    +----------------------------------------------------------+
    |                    LGN Layer                              |
    |   Retinotopic mapping, ON/OFF pathways preserved          |
    +----------------------------------------------------------+
                                |
                                v
    +----------------------------------------------------------+
    |              V1 Layer (Layer 4)                          |
    |   Hypercolumns (orientation columns)                     |
    |   Each hypercolumn = N ensembles with different          |
    |   initial weights -> specialize for different            |
    |   orientations through STDP                              |
    +----------------------------------------------------------+
```

## Key Features

- **Separate ON/OFF pathways**: Retinal ganglion cells with center-surround receptive fields
- **Retinotopic organization**: LGN patches project to V1 hypercolumns
- **STDP learning**: Spike-timing dependent plasticity strengthens coincident activity
- **Lateral inhibition**: Competition between V1 ensembles ensures orientation diversity
- **Conduction delays**: Variable axonal delays for temporal pattern detection
- **GPU acceleration**: Optional CuPy support for faster computation

## Installation

### Requirements
- Python 3.8+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- tqdm >= 4.62.0
- SciPy >= 1.7.0
- CuPy >= 11.0.0 (optional, for GPU acceleration)

### Install dependencies
```bash
pip install -r requirements.txt
```

For GPU support (requires CUDA):
```bash
pip install cupy-cuda11x  # Adjust for your CUDA version
```

## Usage

### Quick Test
```bash
python main.py --test
```
Runs a quick verification of all components.

### Full Training
```bash
python main.py
```
Runs the default training configuration (10 epochs).

### Custom Training
```bash
python main.py --epochs 50 --output my_results
```

### Stimulus Demo Only
```bash
python main.py --demo
```
Generates visualizations of the drifting grating stimuli.

### Force CPU Mode
```bash
python main.py --cpu
```
Disables GPU acceleration even if CuPy is available.

## Configuration

Edit `config.py` to customize:

- `VISUAL_FIELD_SIZE`: Size of visual input (default: 32x32)
- `N_ORIENTATIONS`: Number of orientation stimuli (default: 8)
- `LGN_PATCH_SIZE`: Size of LGN patch per hypercolumn (default: 4x4)
- `N_V1_ENSEMBLES`: Ensembles per hypercolumn (default: 8)
- `STDP_*`: STDP learning parameters
- `N_TRAINING_EPOCHS`: Training duration

## Output

Training generates visualizations in the output directory:

| File | Description |
|------|-------------|
| `weights_heatmap_epochXXX.png` | LGN->V1 weight matrices |
| `tuning_curves_epochXXX.png` | Orientation tuning curves |
| `orientation_map_epochXXX.png` | Preferred orientation map |
| `selectivity_evolution.png` | OSI over training |
| `weight_evolution_ensX.png` | Weight changes over time |
| `training_summary.png` | Comprehensive summary |

## How It Works

### 1. Stimulus Generation
Drifting sinusoidal gratings at various orientations (0, 22.5, 45, ... 157.5 degrees). Each frame is converted to ON/OFF spike trains based on luminance changes.

### 2. RGC -> LGN Processing
ON cells respond to brightness increases, OFF cells to decreases. LGN neurons integrate RGC input with leaky integrate-and-fire dynamics.

### 3. LGN -> V1 Projections
Each V1 hypercolumn receives input from a small LGN patch (e.g., 4x4 neurons). Multiple ensembles within each hypercolumn have different initial weight patterns and conduction delays.

### 4. STDP Learning
- **Pre before post**: Weight increases (LTP)
- **Post before pre**: Weight decreases (LTD)

Over training, ensembles that happen to respond well to a particular orientation get their weights strengthened, while lateral inhibition suppresses competing ensembles.

### 5. Orientation Selectivity Emergence
After training, each ensemble develops a preferred orientation, and the hypercolumn collectively covers all orientations (like a pinwheel structure in V1).

## Metrics

- **OSI (Orientation Selectivity Index)**: (R_pref - R_orth) / (R_pref + R_orth)
  - 0 = no selectivity, 1 = perfect selectivity
- **Orientation Coverage**: Fraction of orientations represented
- **Tuning Width**: Bandwidth of orientation tuning

## File Structure

```
VS1/
├── config.py           # Configuration parameters
├── stimulus.py         # Drifting grating generation
├── neurons.py          # LIF neuron models
├── layers.py           # RGC, LGN, V1 layers
├── learning.py         # STDP and competitive learning
├── visualization.py    # Plotting utilities
├── simulation.py       # Main simulation class
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Example Results

After training, you should see:
- Weight patterns become elongated along preferred orientations
- Tuning curves sharpen with clear peaks
- Different ensembles prefer different orientations
- Mean OSI increases from ~0.1 to ~0.4-0.6

## Troubleshooting

**No GPU acceleration**:
- Ensure CuPy is installed with correct CUDA version
- Check `nvidia-smi` for GPU availability
- Use `--cpu` flag to force CPU mode

**Low orientation selectivity**:
- Increase training epochs
- Adjust STDP parameters (A_plus, A_minus)
- Increase lateral inhibition strength

**Out of memory**:
- Reduce VISUAL_FIELD_SIZE
- Reduce N_V1_ENSEMBLES
- Use CPU mode (slower but less memory)

## References

- Hubel & Wiesel (1962) - Orientation selectivity in cat visual cortex
- Song et al. (2000) - STDP and competitive learning
- Clopath et al. (2010) - Voltage-based STDP

## License

MIT License
