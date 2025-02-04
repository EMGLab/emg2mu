"""
This module provides the main EMG decomposition functionality.
"""

import numpy as np
import warnings
from ..core.preprocessing import whiten, awgn
from ..core.ica import fastICA, torch_fastICA, select_device
from ..detection.duplicate_detection import remove_duplicates, compute_silhouette_scores
from ..utils.io import (load_mat_data, save_results, load_results,
                       save_ica_results, load_ica_results,
                       save_silhouette_scores, load_silhouette_scores)
from ..visualization.plots import plot_spike_train, plot_waveforms


class EMG:
    """
    Class for motor-unit decomposition on hdEMG datasets.

    This class implements the decomposition methodology described in the Hyser Dataset
    by Jian et. al. The original code and dataset is available at PhysioNet.

    Parameters
    ----------
    data : str or numpy.ndarray
        The path to the hdEMG data or the data array itself
    data_mode : str, optional
        EMG recording mode ('monopolar' or 'bipolar'). Default = 'monopolar'
    sampling_frequency : int, optional
        Sampling frequency of the data. Default = 2048 Hz
    extension_parameter : int, optional
        Number of times to repeat data blocks. Default = 4
    max_sources : int, optional
        Maximum iterations for ICA decomposition. Default = 300
    whiten_flag : bool, optional
        Whether to whiten the data prior to ICA. Default = True
    inject_noise : float, optional
        SNR for adding white noise. Default = inf (no noise)
    silhouette_threshold : float, optional
        Threshold for detecting good motor units. Default = 0.6
    output_file : str, optional
        Path for saving results. Default = 'sample_decomposed'
    max_ica_iter : int, optional
        Maximum ICA iterations. Default = 100
    device : str, optional
        Device for torch operations ('auto', 'cuda', 'mps', 'cpu'). Default = 'auto'
    """

    def __init__(self, data, data_mode='monopolar', sampling_frequency=2048,
                 extension_parameter=4, max_sources=300, whiten_flag=True,
                 inject_noise=np.inf, silhouette_threshold=0.6,
                 output_file='sample_decomposed', max_ica_iter=100,
                 device='auto'):

        # Load data
        if isinstance(data, str):
            self.data, self.sampling_frequency = load_mat_data(data)
        else:
            self.data = data
            self.sampling_frequency = sampling_frequency

        # Store parameters
        self.data_mode = data_mode
        self.extension_parameter = extension_parameter
        self.max_sources = max_sources
        self.whiten_flag = whiten_flag
        self.inject_noise = inject_noise
        self.silhouette_threshold = silhouette_threshold
        self.output_file = output_file
        self.max_ica_iter = max_ica_iter
        self.device = select_device(device)

        # Initialize results
        self._preprocessed = None
        self._raw_source = None
        self._raw_spike_train = None
        self._raw_B = None
        self.source = None
        self.spike_train = None
        self.good_idx = None
        self.sil_score = None

    def preprocess(self, array_shape=None):
        """
        Prepare the EMG array for decomposition.

        Parameters
        ----------
        array_shape : list-like, optional
            Shape of the electrode array [rows, cols]. Required for bipolar mode.

        Returns
        -------
        self
            Returns the instance itself for method chaining
        """
        # Get data into column format if needed
        emg = self.data
        num_chan = min(emg.shape)
        if num_chan != emg.shape[1]:
            emg = emg.T

        # Add white noise if specified
        if not np.isinf(self.inject_noise):
            emg = awgn(emg, self.inject_noise)

        # Create bipolar setting from monopolar data if needed
        if self.data_mode == "bipolar":
            if array_shape is None:
                raise ValueError("array_shape is required for bipolar mode")
            for i in range(num_chan - array_shape[0]):
                emg[:, i] = emg[:, i] - emg[:, i + array_shape[0]]
            emg = emg[:, :-array_shape[0]]

        # Extend the data
        extended_emg = np.zeros((emg.shape[0], emg.shape[1] * (self.extension_parameter + 1)))
        extended_emg[:, :emg.shape[1]] = emg
        if self.extension_parameter != 0:
            for i in range(self.extension_parameter):
                i += 1
                extended_emg[i:, emg.shape[1] * i: emg.shape[1] * (i + 1)] = emg[:-i, :]

        # Whiten if requested
        if self.whiten_flag:
            self._preprocessed = whiten(extended_emg)
        else:
            self._preprocessed = extended_emg

        return self

    def run_ica(self, method='fastICA', load_path=None, save_path=None):
        """
        Run ICA decomposition on the preprocessed data.

        Parameters
        ----------
        method : str, optional
            ICA method to use ('fastICA' or 'torch'). Default = 'fastICA'
        load_path : str, optional
            Path to load pre-computed ICA results
        save_path : str, optional
            Path to save ICA results

        Returns
        -------
        self
            Returns the instance itself for method chaining
        """
        if load_path is not None:
            try:
                self._raw_source, self._raw_spike_train, self._raw_B = load_ica_results(load_path)
                return self
            except ValueError as e:
                warnings.warn(f"Could not load ICA results: {str(e)}. Running ICA instead.")

        if self._preprocessed is None:
            raise ValueError("Data must be preprocessed before running ICA")

        if method == 'fastICA':
            self._raw_source, self._raw_B, self._raw_spike_train = fastICA(
                self._preprocessed, self.max_sources, self.max_ica_iter)
        elif method == 'torch':
            self._raw_source, self._raw_B, self._raw_spike_train = torch_fastICA(
                self._preprocessed, self.max_sources, self.max_ica_iter, device=self.device)
        else:
            raise ValueError("method must be either 'fastICA' or 'torch'")

        if save_path is not None:
            save_ica_results(save_path, self._raw_source, self._raw_spike_train, self._raw_B)

        return self

    def remove_duplicates(self, min_firing_rate=4, max_firing_rate=35,
                         max_duplicate_time_diff=0.01, num_bins=50):
        """
        Remove duplicate motor units from the decomposition results.

        Parameters
        ----------
        min_firing_rate : float, optional
            Minimum firing rate in Hz. Default = 4
        max_firing_rate : float, optional
            Maximum firing rate in Hz. Default = 35
        max_duplicate_time_diff : float, optional
            Maximum time difference for duplicates. Default = 0.01
        num_bins : int, optional
            Number of histogram bins. Default = 100

        Returns
        -------
        self
            Returns the instance itself for method chaining
        """
        if self._raw_spike_train is None or self._raw_source is None:
            raise ValueError("ICA must be run before removing duplicates")

        self.spike_train, self.source, self.good_idx = remove_duplicates(
            self._raw_spike_train, self._raw_source, self.sampling_frequency,
            min_firing_rate, max_firing_rate, max_duplicate_time_diff, num_bins)

        return self

    def compute_scores(self, max_samples=1000, load_path=None, save_path=None):
        """
        Compute silhouette scores for the motor units.

        Parameters
        ----------
        max_samples : int, optional
            Maximum samples for score calculation. Default = 1000
        load_path : str, optional
            Path to load pre-computed scores
        save_path : str, optional
            Path to save computed scores

        Returns
        -------
        self
            Returns the instance itself for method chaining
        """
        if load_path is not None:
            try:
                self.sil_score = load_silhouette_scores(load_path)
                return self
            except ValueError as e:
                warnings.warn(f"Could not load scores: {str(e)}. Computing scores instead.")

        if self.spike_train is None or self.source is None:
            raise ValueError("Duplicates must be removed before computing scores")

        self.sil_score = compute_silhouette_scores(
            self.spike_train, self.source, max_samples)

        if save_path is not None:
            save_silhouette_scores(save_path, self.sil_score)

        return self

    def plot(self, plot_type='spike_train', min_score=0.93, **kwargs):
        """
        Plot the decomposition results.

        Parameters
        ----------
        plot_type : str, optional
            Type of plot to generate ('spike_train' or 'waveforms'). Default = 'spike_train'
        min_score : float, optional
            Minimum silhouette score to plot. Default = 0.93
        **kwargs : dict
            Additional keyword arguments passed to the plotting functions

        Returns
        -------
        None
        """
        if self.spike_train is None:
            raise ValueError("No spike train data available to plot")

        if plot_type == 'spike_train':
            plot_spike_train(
                self.spike_train, self.sampling_frequency,
                self.sil_score, min_score, **kwargs)
        elif plot_type == 'waveforms':
            if self.source is None:
                raise ValueError("No source signals available to plot waveforms")
            plot_waveforms(
                self.source, self.spike_train, self.sampling_frequency,
                silhouette_scores=self.sil_score, min_score=min_score, **kwargs)
        else:
            raise ValueError("plot_type must be either 'spike_train' or 'waveforms'")

    def save(self, file_path=None):
        """
        Save the decomposition results.

        Parameters
        ----------
        file_path : str, optional
            Path to save results. If None, uses self.output_file

        Returns
        -------
        None
        """
        path = file_path if file_path is not None else self.output_file
        save_results(path, self.spike_train, self.source,
                    self.good_idx, self.sil_score)

    def load(self, file_path):
        """
        Load previously saved decomposition results.

        Parameters
        ----------
        file_path : str
            Path to the results file

        Returns
        -------
        self
            Returns the instance itself for method chaining
        """
        results = load_results(file_path)
        self.spike_train = results['spike_train']
        self.source = results['source']
        self.good_idx = results['good_idx']
        if 'silhouette_score' in results:
            self.sil_score = results['silhouette_score']
        return self
