# V1 Orientation Selectivity Development

A spiking neural network model of the early visual pathway (RGC → LGN → V1 Layer 4) that develops orientation selectivity through STDP-based learning on drifting gratings.

## Overview

This model implements a biologically-inspired network where:

1. **RGC (Retinal Ganglion Cells)**: ON and OFF channels encode visual stimuli as spike trains
2. **LGN (Lateral Geniculate Nucleus)**: Relay neurons with ON/OFF populations
3. **V1 Layer 4**: Hypercolumns with orientation-selective ensembles that develop through learning

### Key Features

- **Izhikevich neurons** with biologically accurate parameters for different cell types:
  - Regular Spiking (RS) for V1 excitatory neurons
  - Fast Spiking (FS) for V1 inhibitory neurons
  - Thalamocortical (TC) for LGN relay neurons

- **Hypercolumn architecture**: Each hypercolumn receives input from an LGN patch and contains 8 orientation ensembles that compete through lateral inhibition

- **STDP learning**: Spike-timing dependent plasticity shapes the LGN→V1 connections, with different initial conduction delays per orientation ensemble to break symmetry

- **Lateral connectivity**: Excitatory connections between similar orientations, inhibitory connections for competition

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (highly recommended for larger networks):
pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version
```

## Usage

### Basic Training

```bash
# Run with default parameters (3 epochs)
python main.py

# Run with custom epochs
python main.py --epochs 5

# Specify presentations per orientation
python main.py --presentations 20

# Test only (skip training, useful after loading checkpoint)
python main.py --test-only --load checkpoints/epoch_3.pkl
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--presentations` | 10 | Presentations per orientation per epoch |
| `--duration` | 500 | Presentation duration (ms) |
| `--isi` | 100 | Inter-stimulus interval (ms) |
| `--test-only` | False | Skip training, only run testing |
| `--load` | None | Path to checkpoint to load |
| `--seed` | 42 | Random seed |

### Output

Results are saved to the `output/` directory:
- Weight heatmaps for each hypercolumn
- OSI (Orientation Selectivity Index) distributions
- Tuning curves for sample ensembles
- Training progress plots
- Before/after comparison

Checkpoints are saved to `checkpoints/` after each epoch.

## Network Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Visual Stimulus             │
                    │      (Drifting Gratings)            │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │              RGC                     │
                    │   ┌─────────┐  ┌─────────┐         │
                    │   │   ON    │  │   OFF   │         │
                    │   │ 32×32   │  │  32×32  │         │
                    │   └────┬────┘  └────┬────┘         │
                    └────────┼────────────┼───────────────┘
                             │            │
                    ┌────────▼────────────▼───────────────┐
                    │             LGN                      │
                    │   ┌─────────┐  ┌─────────┐         │
                    │   │   ON    │  │   OFF   │         │
                    │   │  TC     │  │   TC    │         │
                    │   │ neurons │  │ neurons │         │
                    │   └────┬────┘  └────┬────┘         │
                    └────────┼────────────┼───────────────┘
                             │            │
                             │   LGN Patch (8×8)
                             │   for each hypercolumn
                             │            │
                    ┌────────▼────────────▼───────────────┐
                    │        V1 Layer 4                    │
                    │                                      │
                    │  Hypercolumn (one per patch)         │
                    │  ┌────────────────────────────┐     │
                    │  │  Ensembles (8 orientations) │     │
                    │  │  ┌──┐ ┌──┐ ┌──┐ ... ┌──┐   │     │
                    │  │  │0°│ │22│ │45│     │157│  │     │
                    │  │  └┬─┘ └┬─┘ └┬─┘     └┬──┘  │     │
                    │  │   │←──excitatory──→│       │     │
                    │  │   │←──inhibitory──→│       │     │
                    │  └────────────────────────────┘     │
                    │                                      │
                    │  × 16 hypercolumns (4×4 grid)        │
                    └──────────────────────────────────────┘
```

## Configuration

Key parameters can be modified in `config.py`:

### Network Dimensions
- `RETINA_SIZE`: Size of visual field (default: 32×32)
- `LGN_PATCH_SIZE`: LGN patch projecting to each hypercolumn (default: 8×8)
- `N_ORIENTATIONS`: Orientations per hypercolumn (default: 8)
- `NEURONS_PER_ENSEMBLE`: Neurons per orientation ensemble (default: 4)

### Connection Parameters
- `W_LGN_V1_INIT_MEAN`: Initial feedforward weights
- `W_LGN_V1_MAX`: Maximum weight (STDP bound)
- `W_V1_EXC_LOCAL`: Lateral excitatory weights
- `W_V1_INH`: Lateral inhibitory weights

### STDP Parameters
- `STDP_TAU_PLUS/MINUS`: Time constants for learning window
- `STDP_A_PLUS/MINUS`: Learning rates

## Understanding the Results

### Orientation Selectivity Index (OSI)

OSI is computed using vector averaging:
```
OSI = |Σ(R × exp(2iθ))| / Σ(R)
```

- **OSI = 0**: No orientation preference (responds equally to all)
- **OSI = 1**: Perfect selectivity (only responds to one orientation)
- **OSI > 0.3**: Typically considered orientation-selective

### What to Look For

1. **Pre-training**: Low OSI (~0.02-0.05), uniform weight distributions
2. **Post-training**: Higher OSI, weight patterns showing oriented structure
3. **Weight heatmaps**: Should show elongated patterns corresponding to orientation

## Biological Plausibility

This model implements several biologically plausible mechanisms:

1. **Feedforward model of simple cells**: Orientation selectivity emerges from aligned ON/OFF LGN inputs, consistent with Hubel & Wiesel's hierarchical model

2. **STDP**: Standard pair-based STDP window based on Bi & Poo (1998)

3. **Lateral competition**: Mexican-hat style connectivity (local excitation, broader inhibition) consistent with cortical organization

4. **Separate ON/OFF channels**: Matches retinal/LGN organization

### Simplifications

- No complex retinal processing (direct spike encoding)
- Single-compartment neurons (no dendritic computation)
- Simplified LGN (no center-surround preprocessing)
- No feedback connections from higher areas

## Troubleshooting

### Low OSI after training
- Increase `--epochs` and `--presentations`
- Increase STDP learning rates (`STDP_A_PLUS/MINUS`)
- Ensure inhibition is strong enough for competition

### Network exploding (very high firing rates)
- Reduce feedforward weights (`W_LGN_V1_INIT_MEAN`)
- Increase inhibitory weights (`W_V1_INH`)
- Reduce lateral excitation (`W_V1_EXC_LOCAL`)

### Slow simulation
- Install CuPy for GPU acceleration
- Reduce `RETINA_SIZE` and `NEURONS_PER_ENSEMBLE`
- Increase `SIM_DT` (less accurate but faster)

## References

- Izhikevich (2003) "Simple Model of Spiking Neurons"
- Bi & Poo (1998) "Synaptic Modifications in Cultured Hippocampal Neurons"
- Srinivasa & Jiang (2013) "Stable learning of functional maps in self-organizing spiking neural networks"
- Hubel & Wiesel (1962) "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex"

## License

MIT License
