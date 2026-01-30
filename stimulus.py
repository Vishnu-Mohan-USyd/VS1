"""
Stimulus generation for drifting gratings.
Generates spiking input from visual stimuli.
"""

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

from config import (
    VISUAL_FIELD_SIZE, GRATING_SPATIAL_FREQ, GRATING_TEMPORAL_FREQ,
    GRATING_CONTRAST, DT, USE_GPU
)


def get_array_module():
    """Get the appropriate array module (CuPy or NumPy)."""
    if USE_GPU and HAS_CUPY:
        return cp
    return np


def to_numpy(arr):
    """Convert array to numpy if it's a CuPy array."""
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def to_device(arr):
    """Convert numpy array to device array if using GPU."""
    xp = get_array_module()
    if xp == cp and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


class DriftingGrating:
    """
    Generates drifting sinusoidal gratings at specified orientations.

    The grating moves perpendicular to its orientation.
    Orientation 0 degrees = vertical bars moving horizontally.
    Orientation 90 degrees = horizontal bars moving vertically.
    """

    def __init__(self, size=VISUAL_FIELD_SIZE, spatial_freq=GRATING_SPATIAL_FREQ,
                 temporal_freq=GRATING_TEMPORAL_FREQ, contrast=GRATING_CONTRAST):
        """
        Initialize the drifting grating generator.

        Args:
            size: Size of the visual field (pixels)
            spatial_freq: Spatial frequency (cycles per pixel)
            temporal_freq: Temporal frequency (Hz)
            contrast: Contrast [0, 1]
        """
        self.size = size
        self.spatial_freq = spatial_freq
        self.temporal_freq = temporal_freq
        self.contrast = contrast

        # Create coordinate grids
        x = np.arange(size) - size / 2
        y = np.arange(size) - size / 2
        self.X, self.Y = np.meshgrid(x, y)

        # Convert to device if using GPU
        self.X = to_device(self.X.astype(np.float32))
        self.Y = to_device(self.Y.astype(np.float32))

    def generate(self, orientation_deg, time_ms):
        """
        Generate a single frame of the drifting grating.

        Args:
            orientation_deg: Orientation in degrees (0 = vertical bars)
            time_ms: Current time in milliseconds

        Returns:
            2D array of luminance values in range [-1, 1]
        """
        xp = get_array_module()

        # Convert orientation to radians
        theta = xp.deg2rad(orientation_deg)

        # Rotate coordinates
        X_rot = self.X * xp.cos(theta) + self.Y * xp.sin(theta)

        # Calculate phase (temporal drift)
        phase = 2 * xp.pi * self.temporal_freq * (time_ms / 1000.0)

        # Generate grating
        grating = self.contrast * xp.sin(2 * xp.pi * self.spatial_freq * X_rot + phase)

        return grating

    def generate_sequence(self, orientation_deg, duration_ms, dt=DT):
        """
        Generate a sequence of grating frames.

        Args:
            orientation_deg: Orientation in degrees
            duration_ms: Total duration in milliseconds
            dt: Time step in milliseconds

        Returns:
            3D array of shape (n_frames, size, size) with values in [-1, 1]
        """
        xp = get_array_module()

        n_frames = int(duration_ms / dt)
        frames = xp.zeros((n_frames, self.size, self.size), dtype=xp.float32)

        for i in range(n_frames):
            t = i * dt
            frames[i] = self.generate(orientation_deg, t)

        return frames


class StimulusEncoder:
    """
    Encodes visual stimuli into spike trains for RGC ON and OFF channels.
    Uses a simple rate coding scheme based on luminance changes.
    """

    def __init__(self, size=VISUAL_FIELD_SIZE, baseline_rate=0.02, gain=0.3):
        """
        Initialize the stimulus encoder.

        Args:
            size: Size of the visual field
            baseline_rate: Baseline firing probability per timestep
            gain: Gain factor for converting luminance to firing rate
        """
        self.size = size
        self.baseline_rate = baseline_rate
        self.gain = gain
        self.prev_frame = None

    def encode(self, frame, use_temporal_diff=True):
        """
        Encode a single frame into ON and OFF spike probabilities.

        ON cells respond to luminance increases (bright regions).
        OFF cells respond to luminance decreases (dark regions).

        Args:
            frame: 2D array of luminance values in range [-1, 1]
            use_temporal_diff: If True, use temporal derivative for encoding

        Returns:
            Tuple of (ON_spikes, OFF_spikes) boolean arrays
        """
        xp = get_array_module()

        if use_temporal_diff and self.prev_frame is not None:
            # Temporal difference encoding
            diff = frame - self.prev_frame

            # ON cells fire when luminance increases
            on_rate = self.baseline_rate + self.gain * xp.maximum(diff, 0)

            # OFF cells fire when luminance decreases
            off_rate = self.baseline_rate + self.gain * xp.maximum(-diff, 0)
        else:
            # Static encoding based on absolute luminance
            # ON cells prefer bright (positive luminance)
            on_rate = self.baseline_rate + self.gain * xp.maximum(frame, 0)

            # OFF cells prefer dark (negative luminance)
            off_rate = self.baseline_rate + self.gain * xp.maximum(-frame, 0)

        # Store frame for next temporal difference
        self.prev_frame = frame.copy()

        # Generate spikes via Poisson process
        on_spikes = xp.random.random(frame.shape) < on_rate
        off_spikes = xp.random.random(frame.shape) < off_rate

        return on_spikes.astype(xp.float32), off_spikes.astype(xp.float32)

    def encode_sequence(self, frames):
        """
        Encode a sequence of frames into spike trains.

        Args:
            frames: 3D array of shape (n_frames, size, size)

        Returns:
            Tuple of (ON_spikes, OFF_spikes) arrays of shape (n_frames, size, size)
        """
        xp = get_array_module()

        n_frames = frames.shape[0]
        on_spikes = xp.zeros_like(frames)
        off_spikes = xp.zeros_like(frames)

        self.prev_frame = None  # Reset for new sequence

        for i in range(n_frames):
            on_spikes[i], off_spikes[i] = self.encode(frames[i])

        return on_spikes, off_spikes

    def reset(self):
        """Reset the encoder state."""
        self.prev_frame = None


def generate_training_batch(orientations, duration_ms, dt=DT, size=VISUAL_FIELD_SIZE):
    """
    Generate a batch of training stimuli for multiple orientations.

    Args:
        orientations: List/array of orientations in degrees
        duration_ms: Duration per stimulus in milliseconds
        dt: Time step in milliseconds
        size: Visual field size

    Returns:
        Dictionary with keys:
            - 'orientations': array of orientations
            - 'on_spikes': list of ON spike trains
            - 'off_spikes': list of OFF spike trains
            - 'frames': list of raw frames (for visualization)
    """
    grating = DriftingGrating(size=size)
    encoder = StimulusEncoder(size=size)

    batch = {
        'orientations': orientations,
        'on_spikes': [],
        'off_spikes': [],
        'frames': []
    }

    for ori in orientations:
        encoder.reset()
        frames = grating.generate_sequence(float(ori), duration_ms, dt)
        on_spikes, off_spikes = encoder.encode_sequence(frames)

        batch['on_spikes'].append(on_spikes)
        batch['off_spikes'].append(off_spikes)
        batch['frames'].append(frames)

    return batch


if __name__ == "__main__":
    # Test stimulus generation
    import matplotlib.pyplot as plt

    print("Testing stimulus generation...")

    grating = DriftingGrating()
    encoder = StimulusEncoder()

    # Generate a short sequence at 45 degrees
    frames = grating.generate_sequence(45, 100, dt=10)
    on_spikes, off_spikes = encoder.encode_sequence(frames)

    # Convert to numpy for plotting
    frames = to_numpy(frames)
    on_spikes = to_numpy(on_spikes)
    off_spikes = to_numpy(off_spikes)

    # Plot first few frames
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    for i in range(4):
        axes[0, i].imshow(frames[i], cmap='gray', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Frame {i}')
        axes[0, i].axis('off')

        axes[1, i].imshow(on_spikes[i], cmap='Reds', vmin=0, vmax=1)
        axes[1, i].set_title(f'ON spikes')
        axes[1, i].axis('off')

        axes[2, i].imshow(off_spikes[i], cmap='Blues', vmin=0, vmax=1)
        axes[2, i].set_title(f'OFF spikes')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('stimulus_test.png', dpi=100)
    plt.close()

    print(f"Generated {len(frames)} frames")
    print(f"ON spike rate: {on_spikes.mean():.4f}")
    print(f"OFF spike rate: {off_spikes.mean():.4f}")
    print("Saved test figure to stimulus_test.png")
