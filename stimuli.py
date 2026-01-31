"""
Visual stimulus generation for the V1 model.

Implements drifting gratings with ON/OFF channel encoding.
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
    RETINA_SIZE, GRATING_SPATIAL_FREQ, GRATING_TEMPORAL_FREQ,
    GRATING_CONTRAST, SPIKE_RATE_MAX, SPIKE_RATE_BASELINE, SIM_DT
)
from neurons import to_numpy, to_gpu


class DriftingGratingGenerator:
    """
    Generates drifting sinusoidal gratings and converts them to ON/OFF spike trains.

    A drifting grating is defined as:
        I(x, y, t) = cos(2*pi*f*(x*cos(theta) + y*sin(theta)) - 2*pi*omega*t)

    where:
        f = spatial frequency (cycles/pixel)
        theta = orientation (radians)
        omega = temporal frequency (Hz)
    """

    def __init__(self, size=RETINA_SIZE, spatial_freq=GRATING_SPATIAL_FREQ,
                 temporal_freq=GRATING_TEMPORAL_FREQ, contrast=GRATING_CONTRAST):
        """
        Initialize the grating generator.

        Args:
            size: Size of the visual field (size x size pixels)
            spatial_freq: Spatial frequency in cycles/pixel
            temporal_freq: Drift speed in Hz
            contrast: Grating contrast (0-1)
        """
        self.size = size
        self.spatial_freq = spatial_freq
        self.temporal_freq = temporal_freq
        self.contrast = contrast

        # Create coordinate grids
        x = np.arange(size) - size / 2
        y = np.arange(size) - size / 2
        self.X, self.Y = np.meshgrid(x, y)
        self.X = to_gpu(self.X.astype(np.float32))
        self.Y = to_gpu(self.Y.astype(np.float32))

        # Current state
        self.current_orientation = 0.0
        self.current_phase = 0.0

    def get_grating(self, orientation_deg, t):
        """
        Get the grating intensity at time t.

        Args:
            orientation_deg: Orientation in degrees (0 = vertical, 90 = horizontal)
            t: Time in ms

        Returns:
            2D array of intensities in range [-1, 1]
        """
        theta = np.radians(orientation_deg)
        omega = self.temporal_freq
        f = self.spatial_freq

        # Compute grating
        # Spatial component: x*cos(theta) + y*sin(theta)
        spatial = self.X * np.cos(theta) + self.Y * np.sin(theta)

        # Full grating with drift (t in ms, convert to seconds)
        phase = 2 * np.pi * f * spatial - 2 * np.pi * omega * (t / 1000.0)
        grating = self.contrast * xp.cos(phase)

        return grating

    def get_on_off_responses(self, orientation_deg, t):
        """
        Convert grating to ON and OFF cell responses.

        ON cells respond to luminance increases (positive values)
        OFF cells respond to luminance decreases (negative values)

        Args:
            orientation_deg: Orientation in degrees
            t: Time in ms

        Returns:
            Tuple of (ON_response, OFF_response), each size x size
        """
        grating = self.get_grating(orientation_deg, t)

        # ON cells: respond to positive (bright)
        # Response = max(0, grating)
        on_response = xp.maximum(grating, 0)

        # OFF cells: respond to negative (dark)
        # Response = max(0, -grating)
        off_response = xp.maximum(-grating, 0)

        return on_response, off_response

    def get_spike_probabilities(self, orientation_deg, t, dt=SIM_DT):
        """
        Convert ON/OFF responses to spike probabilities.

        Uses Poisson spike generation based on instantaneous rate.

        Args:
            orientation_deg: Orientation in degrees
            t: Time in ms
            dt: Time step in ms

        Returns:
            Tuple of (ON_probs, OFF_probs), each size x size
            Values are probabilities of spiking in this time step
        """
        on_resp, off_resp = self.get_on_off_responses(orientation_deg, t)

        # Convert to firing rates
        # Rate = baseline + (max - baseline) * response
        on_rates = SPIKE_RATE_BASELINE + (SPIKE_RATE_MAX - SPIKE_RATE_BASELINE) * on_resp
        off_rates = SPIKE_RATE_BASELINE + (SPIKE_RATE_MAX - SPIKE_RATE_BASELINE) * off_resp

        # Convert rates (Hz) to spike probability per timestep
        # P(spike in dt) = rate * dt / 1000
        on_probs = on_rates * dt / 1000.0
        off_probs = off_rates * dt / 1000.0

        # Clip to valid probability range
        on_probs = xp.clip(on_probs, 0, 1)
        off_probs = xp.clip(off_probs, 0, 1)

        return on_probs, off_probs

    def generate_spikes(self, orientation_deg, t, dt=SIM_DT):
        """
        Generate ON and OFF spikes for current timestep.

        Args:
            orientation_deg: Orientation in degrees
            t: Time in ms
            dt: Time step in ms

        Returns:
            Tuple of (ON_spikes, OFF_spikes), each size x size boolean arrays
        """
        on_probs, off_probs = self.get_spike_probabilities(orientation_deg, t, dt)

        # Poisson spike generation
        on_spikes = xp.random.random(on_probs.shape) < on_probs
        off_spikes = xp.random.random(off_probs.shape) < off_probs

        return on_spikes, off_spikes


class StimulusProtocol:
    """
    Manages the presentation of stimuli during training and testing.
    """

    def __init__(self, orientations, presentation_duration=500, isi=100):
        """
        Initialize stimulus protocol.

        Args:
            orientations: List of orientations to present (degrees)
            presentation_duration: Duration of each presentation (ms)
            isi: Inter-stimulus interval (ms)
        """
        self.orientations = np.array(orientations)
        self.n_orientations = len(orientations)
        self.presentation_duration = presentation_duration
        self.isi = isi
        self.trial_duration = presentation_duration + isi

        # Current state
        self.current_trial = 0
        self.trial_start_time = 0
        self.current_orientation_idx = 0

        # Randomized order for training
        self.trial_order = None
        self.reset()

    def reset(self, shuffle=True):
        """Reset protocol to beginning."""
        self.current_trial = 0
        self.trial_start_time = 0

        if shuffle:
            self.trial_order = np.random.permutation(self.n_orientations)
        else:
            self.trial_order = np.arange(self.n_orientations)

        self.current_orientation_idx = self.trial_order[0]

    def get_current_stimulus(self, t):
        """
        Get current stimulus state at time t.

        Args:
            t: Current time in ms

        Returns:
            Tuple of (orientation_deg, is_stimulus_on)
            orientation_deg is None during ISI
        """
        # Time within current trial
        trial_time = t - self.trial_start_time

        # Check if we need to advance to next trial
        if trial_time >= self.trial_duration:
            self.current_trial += 1
            self.trial_start_time = t
            trial_time = 0

            # Wrap around or get next orientation
            if self.current_trial >= self.n_orientations:
                self.current_trial = 0
                self.trial_order = np.random.permutation(self.n_orientations)

            self.current_orientation_idx = self.trial_order[self.current_trial]

        # Determine if stimulus is on or in ISI
        if trial_time < self.presentation_duration:
            return self.orientations[self.current_orientation_idx], True
        else:
            return None, False

    def get_orientation_sequence(self, n_presentations):
        """
        Generate a sequence of orientations for training.

        Args:
            n_presentations: Total number of presentations

        Returns:
            List of orientations in presentation order
        """
        sequence = []
        for _ in range(n_presentations // self.n_orientations + 1):
            order = np.random.permutation(self.n_orientations)
            sequence.extend(self.orientations[order])

        return sequence[:n_presentations]


def create_blank_stimulus(size=RETINA_SIZE):
    """Create a blank (gray) stimulus that produces baseline activity."""
    on_spikes = xp.zeros((size, size), dtype=xp.bool_)
    off_spikes = xp.zeros((size, size), dtype=xp.bool_)

    # Add some baseline spontaneous activity
    on_spikes = xp.random.random((size, size)) < (SPIKE_RATE_BASELINE * SIM_DT / 1000.0)
    off_spikes = xp.random.random((size, size)) < (SPIKE_RATE_BASELINE * SIM_DT / 1000.0)

    return on_spikes, off_spikes
