"""
Tests for the EMG decomposition module.
"""

import pytest
import numpy as np
import numpy.testing as npt
from emg2mu.core.decomposition import EMG
from emg2mu.tests.test_utils import generate_test_data


def test_emg_initialization():
    """Test EMG class initialization with array data."""
    test_data = generate_test_data()
    emg = EMG(data=test_data['emg_data'])

    assert emg.data.shape == test_data['emg_data'].shape
    assert emg.sampling_frequency == 2048
    assert emg.data_mode == 'monopolar'
    assert emg.extension_parameter == 4
    assert emg.max_sources == 300
    assert emg.whiten_flag is True
    assert np.isinf(emg.inject_noise)
    assert emg.silhouette_threshold == 0.6
    assert emg.output_file == 'sample_decomposed'
    assert emg.max_ica_iter == 100


def test_emg_preprocessing():
    """Test EMG preprocessing with different parameters."""
    test_data = generate_test_data()
    emg = EMG(data=test_data['emg_data'])

    # Test default preprocessing
    emg.preprocess()
    assert emg._preprocessed is not None
    assert emg._preprocessed.shape[1] == test_data['emg_data'].shape[1] * (emg.extension_parameter + 1)

    # Test preprocessing with noise injection
    emg = EMG(data=test_data['emg_data'], inject_noise=20)
    emg.preprocess()
    assert emg._preprocessed is not None

    # Test preprocessing without whitening
    emg = EMG(data=test_data['emg_data'], whiten_flag=False)
    emg.preprocess()
    assert emg._preprocessed is not None


def test_emg_ica():
    """Test ICA decomposition with both CPU and GPU implementations."""
    test_data = generate_test_data()
    emg = EMG(data=test_data['emg_data'], max_sources=4, max_ica_iter=10)
    emg.preprocess()

    # Test CPU implementation
    emg.run_ica(method='fastICA')
    assert emg._raw_source is not None
    assert emg._raw_spike_train is not None
    assert emg._raw_B is not None

    # Test shape of outputs
    assert emg._raw_source.shape[1] == 4  # max_sources
    assert emg._raw_spike_train.shape[1] == 4
    assert emg._raw_B.shape[1] == 4

    # Test PyTorch implementation if available
    try:
        emg.run_ica(method='torch')
        assert emg._raw_source is not None
    except (ImportError, RuntimeError):
        pytest.skip("PyTorch not available or GPU not accessible")


def test_duplicate_removal():
    """Test duplicate motor unit removal."""
    test_data = generate_test_data()
    emg = EMG(data=test_data['emg_data'], max_sources=4, max_ica_iter=10)
    emg.preprocess()
    emg.run_ica()
    emg.remove_duplicates()

    assert emg.spike_train is not None
    assert emg.source is not None
    assert emg.good_idx is not None


def test_silhouette_scores():
    """Test silhouette score computation."""
    test_data = generate_test_data()
    emg = EMG(data=test_data['emg_data'], max_sources=4, max_ica_iter=10)
    emg.preprocess()
    emg.run_ica()
    emg.remove_duplicates()
    emg.compute_scores()

    assert emg.sil_score is not None
    assert len(emg.sil_score) == emg.spike_train.shape[1]
    assert np.all((emg.sil_score >= -1) & (emg.sil_score <= 1))


def test_save_load():
    """Test saving and loading results."""
    import os
    test_data = generate_test_data()
    output_file = 'test_output.npz'

    try:
        emg = EMG(data=test_data['emg_data'], max_sources=4, max_ica_iter=10,
                  output_file=output_file)

        # Process data
        emg.preprocess()
        emg.run_ica()
        emg.remove_duplicates()
        emg.compute_scores()

        # Save results
        emg.save()

        # Verify file exists
        assert os.path.exists(output_file), f"Output file {output_file} not created"

        # Load results in new instance
        emg_loaded = EMG(data=test_data['emg_data'])
        emg_loaded.load(output_file)

        # Compare results
        npt.assert_array_equal(emg.spike_train, emg_loaded.spike_train)
        npt.assert_array_equal(emg.source, emg_loaded.source)
        npt.assert_array_equal(emg.good_idx, emg_loaded.good_idx)
        npt.assert_array_equal(emg.sil_score, emg_loaded.sil_score)

    finally:
        # Clean up test file
        if os.path.exists(output_file):
            os.remove(output_file)


def test_error_handling():
    """Test error handling in EMG class."""
    test_data = generate_test_data()
    emg = EMG(data=test_data['emg_data'])

    # Test running ICA before preprocessing
    with pytest.raises(ValueError):
        emg.run_ica()

    # Test removing duplicates before ICA
    with pytest.raises(ValueError):
        emg.remove_duplicates()

    # Test computing scores before removing duplicates
    with pytest.raises(ValueError):
        emg.compute_scores()

    # Test plotting before processing
    with pytest.raises(ValueError):
        emg.plot()

    # Test invalid ICA method
    emg.preprocess()
    with pytest.raises(ValueError):
        emg.run_ica(method='invalid_method')

    # Test invalid plot type
    with pytest.raises(ValueError):
        emg.plot(plot_type='invalid_type')


def test_bipolar_mode():
    """Test EMG processing in bipolar mode."""
    test_data = generate_test_data()
    emg = EMG(data=test_data['emg_data'], data_mode='bipolar')

    # Test bipolar mode without array shape
    with pytest.raises(ValueError):
        emg.preprocess()

    # Test bipolar mode with array shape
    array_shape = [2, 4]  # 2 rows, 4 columns
    emg.preprocess(array_shape=array_shape)
    assert emg._preprocessed is not None
