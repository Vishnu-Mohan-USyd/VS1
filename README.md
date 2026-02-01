# Biologically Plausible V1 Spiking Network with STDP

A spiking neural network model of the early visual pathway (RGC → LGN → V1 Layer 4) that learns orientation selectivity through spike-timing-dependent plasticity (STDP).

## Network Architecture

```
┌─────────────┐
│   Retina    │  ON and OFF RGC channels (8×8 each)
│  (RGC)      │  Poisson spiking based on grating stimulus
└──────┬──────┘
       │ w_rgc_lgn
       ▼
┌─────────────┐
│    LGN      │  Thalamocortical neurons (128 total: 64 ON + 64 OFF)
│  (TC type)  │  Izhikevich neurons with rebound bursting
└──────┬──────┘
       │ W (plastic, STDP)
       │ with axonal delays D
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    V1 Layer 4                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  8 Excitatory Ensembles (RS type)                   │    │
│  │  E0 ──E1 ──E2 ──E3 ──E4 ──E5 ──E6 ──E7 ─┐          │    │
│  │  └────────────── (ring topology) ────────┘          │    │
│  └─────────────────────────────────────────────────────┘    │
│         │                    │                               │
│         │ E→PV              │ E→SOM                         │
│         ▼                    ▼                               │
│  ┌─────────────┐      ┌─────────────┐                       │
│  │ PV neurons  │      │ SOM neurons │                       │
│  │ (FS type)   │      │ (LTS type)  │                       │
│  │ Local FB    │      │ Lateral     │                       │
│  │ inhibition  │      │ inhibition  │                       │
│  └─────────────┘      └─────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Biological Plausibility Features

### 1. Izhikevich Neurons
Instead of simplified LIF neurons, we use Izhikevich neurons with cell-type-specific parameters from the literature:

| Cell Type | Cortical Equivalent | Parameters (a, b, c, d) |
|-----------|-------------------|-------------------------|
| TC | Thalamocortical (LGN) | (0.02, 0.25, -65, 0.05) |
| RS | Regular Spiking (V1 E) | (0.02, 0.2, -65, 8) |
| FS | Fast Spiking (PV interneurons) | (0.1, 0.2, -65, 2) |
| LTS | Low-Threshold Spiking (SOM interneurons) | (0.02, 0.25, -65, 2) |

References: Izhikevich (2003, 2007)

### 2. Local Inhibitory Circuits
- **PV interneurons (FS)**: Provide fast feedforward inhibition within each ensemble
- **SOM interneurons (LTS)**: Provide lateral inhibition between ensembles

### 3. Triplet STDP
Uses the Pfister & Gerstner (2006) triplet rule with:
- Fast pre/post traces for pair-based updates
- Slow traces for triplet enhancement
- Multiplicative bounds (soft competition)

### 4. Lateral Connectivity with Distance-Dependent Inhibition
The key mechanism for orientation diversity is **distance-dependent lateral inhibition**:

```
Inhibition profile: f(d) = 1 - exp(-d²/(2σ²))
```

This creates a "notch" profile where:
- **Nearby ensembles (small d)**: Weak mutual inhibition → can develop similar preferences
- **Distant ensembles (large d)**: Strong mutual inhibition → pushed to develop different preferences

This mimics the **long-range suppressive interactions** identified by Kaschube et al. (2010) as necessary for universal pinwheel density.

## How Orientation Selectivity Emerges

### The Feedforward Model
Following Miller (1994) and Ferster & Miller (2000), orientation selectivity emerges from the **spatial arrangement of ON/OFF LGN inputs**:

1. **Initial state**: Each V1 ensemble receives random projections from ON and OFF LGN neurons
2. **During learning**: STDP strengthens connections to LGN neurons that consistently fire before the V1 neuron
3. **Result**: Each ensemble develops an elongated receptive field with adjacent ON and OFF subregions

When a grating at the "correct" orientation is presented:
- It activates a line of ON-center LGN neurons and an adjacent line of OFF-center neurons
- These aligned activations sum to drive the V1 neuron
- The neuron fires strongly → high response for preferred orientation

### The Role of Competition
Without competition, all ensembles might develop the same preferred orientation. The SOM-mediated lateral inhibition creates competition:

1. When ensemble E_i fires strongly, its SOM neuron activates
2. The SOM neuron inhibits other ensembles (especially distant ones)
3. Those ensembles fire less, get less STDP potentiation
4. Over time, different ensembles specialize for different orientations

## The Challenge: Achieving Full Orientation Diversity

### What the Model Achieves
- **Orientation selectivity**: OSI increases from ~0.01 (random) to ~0.3-0.4 after training
- **Oriented receptive fields**: ON-OFF weight differences show elongated structure
- **Good diversity**: Typically 5-7 out of 8 orientation bins are covered (improved with new mechanisms)

### Implemented Diversity Mechanisms

The model now includes several biologically-inspired mechanisms for promoting orientation diversity:

#### 1. Retinal Wave Pre-training
Before visual training, the network is exposed to simulated retinal waves - propagating patterns of spontaneous activity that sweep across the retina during early development. This helps establish initial connectivity biases.

```python
# Wave generation
def retinal_wave(direction_deg, t_ms, wave_speed=0.1, wave_width=3.0):
    """Generate propagating wave stimulus"""
```

#### 2. Anti-Hebbian Lateral Inhibition (Foldiak 1990)
SOM→E inhibitory synapses are plastic: ensembles that frequently co-fire develop stronger mutual inhibition, pushing them to respond to different stimuli.

```python
# Rule: ΔW_inh[i,j] = η * y_i * y_j - decay * W_inh[i,j]
```

#### 3. Distance-Dependent Lateral Inhibition
Using a "notch" profile where nearby ensembles have weak mutual inhibition (can develop similar preferences) while distant ensembles have strong inhibition (pushed to differentiate).

```python
# Inhibition profile: f(d) = 1 - exp(-d²/(2σ²))
```

### What Remains Challenging

- **Full coverage**: Some orientation ranges may remain underrepresented depending on random initialization
- **Variability**: Results vary by seed (5-7/8 bins typically achieved)
- **2D topology**: Real V1 is a 2D sheet with pinwheel singularities; our 1D ring is a simplification

### Why This Is Hard

The biological solution to orientation diversity relies on several factors we don't fully capture:

1. **Continuous 2D topology**: Real V1 is a 2D sheet where smooth orientation maps with pinwheel singularities are topological necessities (Wolf & Geisel, 1998)

2. **Multiple timescales**: Biological development occurs over weeks/months with billions of input presentations

3. **Additional plasticity**: Including structural plasticity (synapse formation/elimination) not modeled here

## Usage

```bash
# Basic run with retinal wave pre-training
python biologically_plausible_v1_stdp.py --train-segments 300 --wave-segments 50 --seed 42

# Skip wave pre-training
python biologically_plausible_v1_stdp.py --train-segments 300 --wave-segments 0 --seed 42

# Adjust lateral inhibition notch profile
python biologically_plausible_v1_stdp.py --som-notch-sigma 1.5

# Different output directory
python biologically_plausible_v1_stdp.py --out runs/experiment1
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train-segments` | 200 | Number of visual training segments |
| `--wave-segments` | 50 | Number of retinal wave pre-training segments |
| `--segment-ms` | 300 | Duration of each segment |
| `--N` | 8 | LGN patch size (N×N) |
| `--M` | 8 | Number of V1 ensembles |
| `--som-notch-sigma` | 1.0 | Width of inhibition notch |
| `--coord-jitter` | 0.25 | RGC position jitter |

## Example Results

Running with wave pre-training and visual training (seed=42):

```
[baseline]   OSI=0.008, pref_bins=6/8
[post-wave]  OSI=0.168, pref_bins=6/8
[seg 150]    OSI=0.367, pref_bins=6/8
[seg 300]    OSI=0.300, pref_bins=5/8
[final]      OSI=0.317, pref_bins=6/8
```

Results across different seeds (300 training segments + 50 wave segments):

| Seed | Final OSI | Bins Covered | Notes |
|------|-----------|--------------|-------|
| 42   | 0.317     | 6/8          | Good balance |
| 123  | 0.314     | 7/8          | Best diversity |
| 456  | 0.387     | 5/8          | Highest OSI |
| 789  | 0.341     | 5/8          | Good OSI |

The network learns orientation selectivity (OSI increases ~30-40x from baseline) with improved diversity (5-7/8 bins covered) using the new mechanisms.

## References

- **Izhikevich (2003)** - Simple model of spiking neurons. IEEE Trans Neural Networks
- **Izhikevich (2007)** - Dynamical Systems in Neuroscience. MIT Press
- **Miller (1994)** - A model for the development of simple cell receptive fields. J Neurosci
- **Ferster & Miller (2000)** - Neural mechanisms of orientation selectivity. Annu Rev Neurosci
- **Pfister & Gerstner (2006)** - Triplets of spikes in a model of spike timing-dependent plasticity. J Neurosci
- **Kaschube et al. (2010)** - Universality in the evolution of orientation columns. Science
- **Wolf & Geisel (1998)** - Spontaneous pinwheel annihilation during visual development. Nature
- **Swindale (1996)** - The development of topography in visual cortex. Network
- **Hansel & van Vreeswijk (2012)** - The mechanism of orientation selectivity without a functional map. J Neurosci
- **Vogels et al. (2011)** - Inhibitory plasticity balances excitation and inhibition. Science
- **Foldiak (1990)** - Forming sparse representations by local anti-Hebbian learning. Biol Cybern
- **Butts (2002)** - Retinal waves: implications for synaptic learning rules during development. Neuroscientist

## Comparison with Biology

| Feature | This Model | Biology |
|---------|------------|---------|
| LGN neurons | Izhikevich TC | Real TC neurons with intrinsic rhythms |
| ON/OFF channels | Separate populations | Anatomically segregated, with distinct spatial organization |
| LGN→V1 weights | STDP-plastic | Multiple plasticity mechanisms |
| Orientation selectivity | Emerges from STDP | Emerges from feedforward + recurrent mechanisms |
| Lateral inhibition | SOM-mediated, distance-dependent + anti-Hebbian | Multiple interneuron types, feature-specific |
| Map organization | 1D ring of 8 ensembles | 2D sheet with pinwheels |
| Development | Retinal waves → visual training | Spontaneous waves → visual experience |

## License

MIT
