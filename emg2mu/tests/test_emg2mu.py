import os.path as op
import numpy.testing as npt
import emg2mu as ece
import numpy as np

data_path = op.join(ece.__path__[0], 'data')
#Load data like: op.join(data_path, 'mydatafile.dat')


def test_trivial():
    """
    Should always pass. Just used to ensure that py.test is setup correctly.
    """
    npt.assert_equal(np.array([1, 1, 1]), np.array([1, 1, 1]))


def generate_fake_emg_data():
    """
    Generate fake 8-channel EMG data, 8 sources, and corresponding spike train for testing.
    """
    np.random.seed(0)
    emg_data = np.random.randn(1000, 8)
    sources = np.random.randn(1000, 8)
    spike_train = (np.random.rand(1000, 8) > 0.95).astype(int)
    return emg_data, sources, spike_train


def test_spikeTrain_plot_color():
    """
    Test the spikeTrain_plot method with color_plot=True.
    """
    emg_data, sources, spike_train = generate_fake_emg_data()
    emg = ece.EMG(data=emg_data)
    emg.spike_train = spike_train
    emg.sil_score = np.random.rand(8)
    emg.spikeTrain_plot(color_plot=True)
    # Add assertions to verify the plot if needed


def test_spikeTrain_plot_monochrome():
    """
    Test the spikeTrain_plot method with color_plot=False.
    """
    emg_data, sources, spike_train = generate_fake_emg_data()
    emg = ece.EMG(data=emg_data)
    emg.spike_train = spike_train
    emg.sil_score = np.random.rand(8)
    emg.spikeTrain_plot(color_plot=False)
    # Add assertions to verify the plot if needed


def test_spikeTrain_plot_spike_height():
    """
    Test the spikeTrain_plot method with different spike_height values.
    """
    emg_data, sources, spike_train = generate_fake_emg_data()
    emg = ece.EMG(data=emg_data)
    emg.spike_train = spike_train
    emg.sil_score = np.random.rand(8)
    emg.spikeTrain_plot(spike_height=0.1)
    emg.spikeTrain_plot(spike_height=0.3)
    # Add assertions to verify the plot if needed


def test_spikeTrain_plot_spike_length():
    """
    Test the spikeTrain_plot method with different spike_length values.
    """
    emg_data, sources, spike_train = generate_fake_emg_data()
    emg = ece.EMG(data=emg_data)
    emg.spike_train = spike_train
    emg.sil_score = np.random.rand(8)
    emg.spikeTrain_plot(spike_length=0.005)
    emg.spikeTrain_plot(spike_length=0.02)
    # Add assertions to verify the plot if needed
