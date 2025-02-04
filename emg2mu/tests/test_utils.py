"""
Utility functions for testing EMG decomposition.
"""

import numpy as np
from scipy import signal


def gaussian_pulse(length, std):
    """Generate a Gaussian pulse."""
    return signal.windows.gaussian(length, std=std)


def generate_synthetic_emg(n_samples=1000, n_channels=8, sampling_freq=2048,
                         noise_level=0.1, seed=42):
    """
    Generate synthetic EMG data for testing.

    Parameters
    ----------
    n_samples : int
        Number of time samples
    n_channels : int
        Number of EMG channels
    sampling_freq : int
        Sampling frequency in Hz
    noise_level : float
        Level of Gaussian noise to add
    seed : int
        Random seed for reproducibility

    Returns
    -------
    emg_data : numpy.ndarray
        Synthetic EMG data of shape (n_samples, n_channels)
    sources : numpy.ndarray
        Ground truth source signals
    spike_train : numpy.ndarray
        Ground truth spike train
    """
    np.random.seed(seed)

    # Generate source signals (motor unit action potentials)
    sources = np.zeros((n_samples, n_channels))
    spike_train = np.zeros((n_samples, n_channels))

    for i in range(n_channels):
        # Generate random firing times (spikes)
        firing_rate = np.random.uniform(8, 20)  # Hz
        spike_times = np.random.uniform(0, n_samples / sampling_freq,
                                      int(firing_rate * n_samples / sampling_freq))
        spike_indices = (spike_times * sampling_freq).astype(int)
        spike_indices = spike_indices[spike_indices < n_samples]
        spike_train[spike_indices, i] = 1

        # Generate MUAP waveform
        duration = 0.01  # 10ms
        t_muap = np.linspace(-duration / 2, duration / 2, int(duration * sampling_freq))
        muap = gaussian_pulse(len(t_muap), std=len(t_muap) / 8)

        # Convolve spikes with MUAP
        sources[:, i] = np.convolve(spike_train[:, i], muap, mode='same')

    # Mix sources
    mixing_matrix = np.random.randn(n_channels, n_channels)
    emg_data = np.dot(sources, mixing_matrix.T)

    # Add noise
    noise = np.random.randn(*emg_data.shape) * noise_level
    emg_data += noise

    return emg_data, sources, spike_train


def generate_test_data():
    """
    Generate a standard set of test data.

    Returns
    -------
    dict
        Dictionary containing test data and parameters
    """
    emg_data, sources, spike_train = generate_synthetic_emg()

    return {
        'emg_data': emg_data,
        'sources': sources,
        'spike_train': spike_train,
        'sampling_frequency': 2048,
        'n_channels': 8,
        'n_samples': 1000
    }
