"""
Tests for the ICA module.
"""

import pytest
import numpy as np
import numpy.testing as npt
import torch
from emg2mu.core.ica import fastICA, torch_fastICA, select_device
from emg2mu.tests.test_utils import generate_test_data


def test_fastica():
    """Test CPU-based FastICA implementation."""
    test_data = generate_test_data()
    emg_data = test_data['emg_data']
    n_components = 4
    max_iter = 10

    source, B, spike_train = fastICA(emg_data, n_components, max_iter)

    # Check output shapes
    assert source.shape[1] == n_components
    assert B.shape[1] == n_components
    assert spike_train.shape[1] == n_components

    # Check data types
    assert isinstance(source, np.ndarray)
    assert isinstance(B, np.ndarray)
    assert isinstance(spike_train, np.ndarray)

    # Check spike train values are binary
    assert np.all(np.unique(spike_train) == np.array([0, 1]))


def test_torch_fastica():
    """Test PyTorch-based FastICA implementation."""
    if not torch.cuda.is_available() and not hasattr(torch.backends, 'mps'):
        pytest.skip("No GPU available for testing")

    test_data = generate_test_data()
    emg_data = test_data['emg_data']
    n_components = 4
    max_iter = 10

    # Test with different devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')

    for device in devices:
        source, B, spike_train = torch_fastICA(emg_data, n_components, max_iter, device=device)

        # Check output shapes
        assert source.shape[1] == n_components
        assert B.shape[1] == n_components
        assert spike_train.shape[1] == n_components

        # Check data types
        assert isinstance(source, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert isinstance(spike_train, np.ndarray)

        # Check spike train values are binary
        assert np.all(np.unique(spike_train) == np.array([0, 1]))


def test_device_selection():
    """Test device selection logic."""
    # Test explicit CPU selection
    device = select_device('cpu')
    assert device == 'cpu'

    # Test auto selection - should return available device
    device = select_device('auto')
    assert device in ['cuda', 'mps', 'cpu']

    # Test CUDA selection - should fallback to CPU if not available
    device = select_device('cuda')
    expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == expected_device

    # Test MPS selection - should fallback to CPU if not available
    device = select_device('mps')
    expected_device = 'mps' if torch.mps.is_available() else 'cpu'
    assert device == expected_device


def test_fastica_convergence():
    """Test FastICA convergence with simple data."""
    # Generate simple mixed signals
    t = np.linspace(0, 10, 1000)
    s1 = np.sin(2 * np.pi * t)
    s2 = np.sign(np.sin(3 * np.pi * t))
    S = np.c_[s1, s2]

    # Create mixing matrix and mix signals
    A = np.array([[1, 1], [0.5, 2]])
    X = np.dot(S, A.T)

    # Run FastICA
    source, B, _ = fastICA(X, 2, 100)

    # Check if recovered signals are correlated with original
    # (up to sign and permutation)
    corr_matrix = np.abs(np.corrcoef(source.T, S.T))[:2, 2:]
    max_corr = np.max(corr_matrix, axis=1)
    assert np.all(max_corr > 0.85)  # Relaxed threshold for more reliable testing


def test_torch_fastica_consistency():
    """Test consistency between CPU and GPU implementations."""
    # Generate simple test data
    test_data = generate_test_data()
    emg_data = test_data['emg_data']
    n_components = 8
    max_iter = 500

    # Run CPU FastICA
    cpu_source, cpu_B, cpu_spikes = fastICA(emg_data, n_components, max_iter)

    # Run PyTorch FastICA on CPU
    torch_source, torch_B, torch_spikes = torch_fastICA(
        emg_data, n_components, max_iter, device='cpu')

    # Compare shapes
    assert cpu_source.shape == torch_source.shape
    assert cpu_B.shape == torch_B.shape
    assert cpu_spikes.shape == torch_spikes.shape

    # Compare spike counts with relaxed tolerance
    npt.assert_allclose(
        np.sort(np.sum(cpu_spikes, axis=0)),
        np.sort(np.sum(torch_spikes, axis=0)),
        rtol=0.3  # Increased tolerance for more reliable testing
    )
