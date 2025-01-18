"""
emg2mu.py

This module provides functions for motor-unit decomposition on hdEMG datasets.
It includes functions for adding white Gaussian noise to a signal, whitening a matrix,
and computing the mean Silhouette Coefficient of all samples.

(c) Seyed Yahya Shirazi, SCCN, 2023-2025

Copyright:
This module is part of the emg2mu package, released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License.
You should have received a copy of the license along with the emg2mu package.
If not, see https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import warnings


def awgn(sig, reqSNR, *args):
    """
    Add white Gaussian noise to a signal.

    Parameters
    ----------
    sig : numpy.ndarray
        The input signal.
    reqSNR : float
        The desired signal-to-noise ratio in dB.
    *args : Union[str, float], optional
        Additional arguments to specify the signal power and power type.
        If specified, the first argument must be either a scalar representing the signal power in dBW
        or the string 'measured' to indicate that the function should measure the signal power.
        If a scalar is provided, the second argument (optional) must be either 'db' or 'linear'
        to specify the units of the signal power and SNR. If not specified, the default is 'db'.

    Returns
    -------
    y : numpy.ndarray
        The noisy signal.

    Raises
    ------
    ValueError
        If the signal input is not a non-empty numpy array.
        If the SNR input is not a real scalar greater than 0.
        If the signal power input is not a real scalar greater than 0.
        If the signal power must be positive for linear scale.
        If the SNR must be positive for linear scale.

    Warnings
    --------
    UserWarning
        If the third or fourth argument is not 'db' or 'linear'.
    """
    # Validate signal input
    if not isinstance(sig, np.ndarray) or sig.size == 0:
        raise ValueError("The signal input must be a non-empty numpy array.")

    # Validate SNR input
    if not isinstance(reqSNR, (int, float)) or reqSNR <= 0:
        raise ValueError("The SNR input must be a real scalar greater than 0.")

    # Validate signal power
    if len(args) >= 1:
        if isinstance(args[0], str) and args[0].lower() == 'measured':
            sigPower = np.sum(np.abs(sig)**2) / sig.size
        else:
            sigPower = args[0]
            if not isinstance(sigPower, (int, float)) or sigPower <= 0:
                raise ValueError("The signal power input must be a real scalar greater than 0.")
    else:
        sigPower = 1

    # Validate power type
    isLinearScale = False
    if len(args) >= 2:
        if isinstance(args[1], str) and args[1].lower() in ['db', 'linear']:
            isLinearScale = args[1].lower() == 'linear'
        else:
            warnings.warn("The third argument must either be 'db' or 'linear'.")

    if len(args) == 3:
        if isinstance(args[2], str) and args[2].lower() in ['db', 'linear']:
            isLinearScale = args[2].lower() == 'linear'
        else:
            warnings.warn("The fourth argument must either be 'db' or 'linear'.")

    # Convert signal power and SNR to linear scale
    if not isLinearScale:
        if len(args) >= 1 and not isinstance(args[0], str):
            sigPower = 10**(sigPower / 10)
        reqSNR = 10**(reqSNR / 10)

    # Check for invalid signal power and SNR for linear scale
    if isLinearScale and sigPower <= 0:
        raise ValueError("The signal power must be positive for linear scale.")
    if isLinearScale and reqSNR <= 0:
        raise ValueError("The SNR must be positive for linear scale.")

    noisePower = sigPower / reqSNR

    # Add noise
    if np.iscomplexobj(sig):
        noise = np.sqrt(noisePower / 2) * (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape))
    else:
        noise = np.sqrt(noisePower) * np.random.randn(*sig.shape)
    y = sig + noise
    return y


def whiten(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    Outputs:
        X_hat:  Whitened data matrix

    References:
    https://gist.github.com/joelouismarino/ce239b5601fff2698895f48003f7464b
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None
    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
    if method == 'zca':
        W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
    elif method == 'pca':
        W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
    elif method == 'cholesky':
        W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')
    return np.dot(X_centered, W.T)


class EMG:
    """
    # motor-unit decomposition on hdEMG datasets

    This function and the helper files are mainly a python implementation of the code accompanied with
    Hyser Dataset by Jian et. al.
    The original code and the dataset is also available at PhysioNet

    Parameters
    ----------
    data : str or numpy.ndarray
        The path to the hdEMG data. If 'data' is pointing to the location of the data file, the data file must be
        a MAT array.
        Default is the sample file included in the toolbox.
    data_mode : str, optional
        EMG can be recorded in the 'monopolar' or 'bipolar' mode. Default = 'monopolar'
    sampling_frequency : int, optional
        Sampling frequency of the data. Default = 2048 Hz
    extension_parameter : int, optional
        The number of times to repeat the data blocks, see the Hyser paper for more detail. Default = 4
    max_sources : int, optional
        Maximum iterations for the (FAST) ICA decomposition. Default = 300
    whiten_flag : int, optional
        Whether to whiten the data prior to the ICA. Default = 1
    inject_noise : float, optional
        Adding white noise to the EMG mixture. Uses the equivalent of the Communication Toolbox AWGN function.
        Default = Inf for not injecting any artificial noise.
    silhouette_threshold : float, optional
        The silhouette threshold to detect the good motor units. Default = 0.6
    output_file : str, optional
        The path where the files should be saved. The function does not create the path, rather uses it.
        Default is the 'sample' path of the toolbox.
    max_ica_iter : int, optional
        Maximum number of iterations for ICA algorithm. Default = 100
    plot_spikeTrain : bool, optional
        Whether to plot the resulting spike trains. Default = True
    load_ICA : bool, optional
        Whether to load precomputed ICA results. Default = False
    save_ICA : bool, optional
        Whether to save ICA results for later use. Default = False
    ICA_path : str, optional
        Path to save/load ICA results. Default = None
    load_score : bool, optional
        Whether to load precomputed silhouette scores. Default = False
    save_score : bool, optional
        Whether to save silhouette scores for later use. Default = False
    score_path : str, optional
        Path to save/load silhouette scores. Default = None
    num_bins : int, optional
        Number of bins used for histogram analysis in duplicate detection. Default = 100
    min_firing_rate : float, optional
        Minimum firing rate in Hz for valid motor units. Default = 4
    max_firing_rate : float, optional
        Maximum firing rate in Hz for valid motor units. Default = 35
    max_duplicate_time_diff : float, optional
        Maximum time difference for considering motor units as duplicates. Default = 0.01
    device : str, optional
        Device to use for torch operations. Can be 'auto', 'cuda', 'mps', 'cpu', or specific CUDA device.
        'auto' will automatically select the best available device. Default = 'auto'

    Returns
    -------
    motor_unit : dict
        The structure including the following fields:
        spike_train : Motor unit spike train data
        source : Source signals from ICA decomposition
        good_idx : Indices of valid motor units
        silhouette_score : Silhouette scores for motor units
    """

    def __init__(self, data, data_mode='monopolar', sampling_frequency=2048, extension_parameter=4, max_sources=300,
                 whiten_flag=1, inject_noise=np.inf, silhouette_threshold=0.6, output_file='sample_decomposed',
                 max_ica_iter=100, plot_spikeTrain=1,
                 load_ICA=False, save_ICA=False, ICA_path=None, load_score=False, save_score=False, score_path=None,
                 num_bins=100, min_firing_rate=4, max_firing_rate=35, max_duplicate_time_diff=0.01,
                 device='auto', max_silhouette_samples=1000, ica_tolerance=1e-5,
                 plot_target_height=800, plot_units_per_height=40):
        # Processing parameters
        self.num_bins = num_bins  # Number of bins for histogram in duplicate detection
        self.min_firing_rate = min_firing_rate  # Minimum firing rate in Hz
        self.max_firing_rate = max_firing_rate  # Maximum firing rate in Hz
        self.max_duplicate_time_diff = max_duplicate_time_diff  # Maximum time difference for duplicate detection
        self.max_silhouette_samples = max_silhouette_samples  # Maximum samples for silhouette calculation
        self.ica_tolerance = ica_tolerance  # Convergence tolerance for ICA

        # Plot parameters
        self.plot_target_height = plot_target_height  # Target plot height in pixels
        self.plot_units_per_height = plot_units_per_height  # Number of units per height unit

        # Device selection for torch operations
        self.device = self._select_device(device)

        if isinstance(data, str):
            try:
                import scipy.io as sio
                emg_file = sio.loadmat(data)
                self.data = emg_file['Data'][0, 0]
                self.sampling_frequency = emg_file['SamplingFrequency'].item()  # only get the element, not the array
            except ImportError:
                raise ValueError("The data file must be a MAT array.")
        else:
            self.data = data
            self.sampling_frequency = sampling_frequency
        self.data_mode = data_mode
        self.extension_parameter = extension_parameter
        self.max_sources = max_sources
        self.whiten_flag = whiten_flag
        self.inject_noise = inject_noise
        self.silhouette_threshold = silhouette_threshold
        self.output_file = output_file
        self.max_ica_iter = max_ica_iter
        self.plot_spikeTrain = plot_spikeTrain
        self.load_ICA = load_ICA
        self.save_ICA = save_ICA
        self.ICA_path = ICA_path
        self.load_score = load_score
        self.save_score = save_score
        self.score_path = score_path
        self.motor_unit = {}

    def preprocess(self, data_mode='monopolar', whiten_flag=True, R=4, SNR=np.inf, array_shape=[8, 8]):
        '''
        Prepares the emg array for decomposition.

        Parameters
        ----------
        data : array-like
            The hd-EMG data
        data_mode : str
            EMG can be recorded in the 'monopolar' or 'bipolar' mode. Default = 'monopolar'
        whiten_flag : bool
            Whether to whiten the data prior to the ICA. Default = 1
        SNR : float
            Adding white noise to the EMG mixture. Uses the equivalnet of Matlab's Communication Toolbox AWGN function.
            Default = Inf for not injecting any artificial noise.
        R : int
            The number of times to repeat the data blocks, see the Hyser paper for more detail. Default = 4
        array_shape : list-like
            The first element will be used to calculate the bipolar activity if the bipolar flag is
            on for the 'data_mode'.

        Returns
        -------
        preprocessed_data : array-like
            The preprocessed EMG data
        W : array-like
            The whitening matrix used for preprocessing
        '''
        # data can come in column or row format, but needs to become the column format where the
        emg = self.data
        num_chan = min(emg.shape)  # Let's assume that we have more than 64 frames
        if num_chan != emg.shape[1]:
            emg = emg.T

        # Add white noise
        if not np.isinf(SNR):
            emg = awgn(emg, SNR, 'dB')

        # create a bipolar setting from the monopolar data
        if data_mode == "bipolar":
            for i in range(num_chan - array_shape[0]):
                emg[:, i] = emg[:, i] - emg[:, i + array_shape[0]]
            emg = emg[:, :-8]

        extended_emg = np.zeros((emg.shape[0], emg.shape[1] * (R + 1)))
        extended_emg[:, :emg.shape[1]] = emg  # to make a consistent downstream
        if R != 0:
            for i in range(R):
                # This basically shifts the data on the replications one step
                # forward. This is pretty standard in the ICA, as ICA should be
                # able to parse the sources out pretty well. Also, it introduces
                # small delay, R/freq, which with R=64, delay = 64/2048= 31ms.
                # This addition reinforces finding MUAPs, despite having
                # duplicates. Later on, the duplicates will be removed.
                i += 1
                extended_emg[i:, emg.shape[1] * i: emg.shape[1] * (i + 1)] = emg[:-i, :]

        if whiten_flag:
            whitened_data = whiten(extended_emg)
            preprocessed_emg = whitened_data
        else:
            preprocessed_emg = extended_emg

        self._preprocessed = preprocessed_emg
        # return self

    def _select_device(self, device_preference='auto'):
        """
        Select the appropriate device for torch operations.

        Parameters
        ----------
        device_preference : str
            Can be 'auto', 'cuda', 'mps', 'cpu', or a specific CUDA device like 'cuda:0'

        Returns
        -------
        str
            The selected device string for torch
        """
        import torch

        if device_preference != 'auto':
            return device_preference

        # Check for CUDA
        if torch.cuda.is_available():
            return 'cuda'
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        # Fallback to CPU
        return 'cpu'

    def _torch_fastICA_(self, extended_emg, M, max_iter, tolerance=None, device=None):
        """
        Run the ICA decomposition using torch

        Parameters
        ----------
        extended_emg : numpy.ndarray
            The preprocessed extended EMG data
        M : int
            Maximum number of sources being decomposed by (FAST) ICA
        max_iter : int
            Maximum iterations for the (FAST) ICA decompsition
        tolerance : float, optional
            Convergence tolerance for ICA. If None, uses class-level ica_tolerance.

        Returns
        -------
        uncleaned_source : numpy.ndarray
            The uncleaned sources from the ICA decomposition
        B : numpy.ndarray
            The unmixing matrix
        uncleaned_spkieTrain : numpy.ndarray
            The uncleaned spike train
        score : numpy.ndarray
            The score of the sources
        """
        import torch
        import torch.nn.functional as F
        from scipy.signal import find_peaks
        from sklearn.cluster import KMeans

        _tolerance = tolerance if tolerance is not None else self.ica_tolerance
        _device = device if device is not None else self.device

        emg = torch.tensor(extended_emg.T, dtype=torch.float32, device=_device)
        num_chan, frames = emg.shape
        B = torch.zeros((num_chan, M), dtype=torch.float32, device=_device)
        spike_train = torch.zeros((frames, M), dtype=torch.float32, device=_device)
        source = torch.zeros((frames, M), dtype=torch.float32, device=_device)
        # score = torch.zeros(M, dtype=torch.float32, device=device)
        print(f"running ICA for {M} sources")
        for i in range(M):
            w = []
            w.append(torch.randn(num_chan, 1, device=_device, dtype=torch.float32))
            w.append(torch.randn(num_chan, 1, device=_device, dtype=torch.float32))
            for n in range(1, max_iter):
                if abs(torch.matmul(w[n].T, w[n - 1]) - 1) > _tolerance:
                    A = torch.mean(2 * torch.matmul(w[n].T, emg))
                    w.append(emg @ ((torch.matmul(w[n].T, emg).T) ** 2) - A * w[n])
                    w[-1] = w[-1] - torch.matmul(torch.matmul(B, B.T), w[-1])
                    w[-1] = F.normalize(w[-1], p=2, dim=0)
                else:
                    break
            source[:, i] = torch.matmul(w[-1].T, emg)
            pow = torch.pow(source[:, i], 2)
            loc, _ = find_peaks(pow.cpu().numpy())
            pks = pow[loc].cpu().numpy()
            kmeans = KMeans(n_clusters=2, n_init=10).fit(pks.reshape(-1, 1))
            idx = kmeans.labels_
            # score[i] = silhouette_score_torch(torch.tensor(pks.reshape(-1, 1),
            #                                              dtype=torch.float32, device=device),idx)
            if sum(idx == 1) <= sum(idx == 0):
                spike_loc = loc[idx == 1]
            else:
                spike_loc = loc[idx == 0]
            spike_train[spike_loc, i] = 1
            B[:, i] = w[-1].flatten()
            print(".", end="")
            if i > 1 and (i - 1) % 50 == 0:
                print('\n')
        spike_train = spike_train.cpu().numpy()
        print("\nICA decomposition is completed")
        return source.cpu().numpy(), B.cpu().numpy(), spike_train

    def _fastICA_(self, extended_emg, M, max_iter, tolerance=None):
        """
        Run the ICA decomposition

        Parameters
        ----------
        extended_emg : numpy.ndarray
            The preprocessed extended EMG data
        M : int
            Maximum number of sources being decomposed by (FAST) ICA
        max_iter : int
            Maximum iterations for the (FAST) ICA decompsition
        tolerance : float, optional
            Convergence tolerance for ICA. If None, uses class-level ica_tolerance.

        Returns
        -------
        uncleaned_source : numpy.ndarray
            The uncleaned sources from the ICA decomposition
        B : numpy.ndarray
            The unmixing matrix
        uncleaned_spkieTrain : numpy.ndarray
            The uncleaned spike train
        """
        from scipy.signal import find_peaks
        from sklearn.cluster import KMeans
        _tolerance = tolerance if tolerance is not None else self.ica_tolerance
        emg = extended_emg.T
        num_chan, frames = emg.shape
        B = np.zeros((num_chan, M))
        spike_train = np.zeros((frames, M))
        source = np.zeros((frames, M))
        print(f"running ICA for {M} sources")
        for i in range(M):
            # print(f"running ICA for source number {i + 1}")
            w = []
            w.append(np.random.randn(num_chan, 1))
            w.append(np.random.randn(num_chan, 1))
            for n in range(1, max_iter):
                if abs(np.dot(w[n].T, w[n - 1]) - 1) > _tolerance:
                    A = np.mean(2 * np.dot(w[n].T, emg))
                    w.append(emg @ ((np.dot(w[n].T, emg).T) ** 2) - A * w[n])
                    w[-1] = w[-1] - np.dot(np.dot(B, B.T), w[-1])
                    w[-1] = w[-1] / np.linalg.norm(w[-1])
                else:
                    break
            source[:, i] = np.dot(w[-1].T, emg)
            pow = np.power(source[:, i], 2)
            loc, _ = find_peaks(pow)
            pks = pow[loc]
            kmeans = KMeans(n_clusters=2, n_init=10).fit(pks.reshape(-1, 1))
            idx = kmeans.labels_
            # score[i] = silhouette_score(pks.reshape(-1, 1), idx)
            if sum(idx == 0) <= sum(idx == 1):
                spike_loc = loc[idx == 0]
            else:
                spike_loc = loc[idx == 1]
            spike_train[spike_loc, i] = 1
            B[:, i] = w[-1].flatten()
            print(".", end="")
            if i > 1 and (i - 1) % 50 == 0:
                print("\n")
        spike_train = np.array(spike_train)
        print("\nICA decomposition is completed")
        return source, B, spike_train

    def run_ICA(self, method='fastICA', device=None, max_iter=None, tolerance=None):
        """
        Run the ICA algorithm

        Parameters
        ----------
        method : str
            The ICA algorithm to be used. Either 'fastICA' or 'torch'
        device : str, optional
            Device to use for torch operations. If None, uses the class-level device setting.
            Can be 'auto', 'cuda', 'mps', 'cpu', or specific CUDA device.
        max_iter : int, optional
            Maximum iterations for ICA algorithm. If None, uses the class-level max_ica_iter setting.
        tolerance : float, optional
            Convergence tolerance for ICA. If None, uses the class-level ica_tolerance setting.

        Attributes
        ----------
        _raw_source : numpy.ndarray
            The uncleaned sources from the ICA decomposition
        _raw_spike_train : numpy.ndarray
            The uncleaned spike train
        _raw_B : numpy.ndarray
            The unmixing matrix
        """
        if self.load_ICA:
            try:
                self._load_ICA_(self.ICA_path)
                return
            except FileNotFoundError:
                print(f"ICA results not found at {self.ICA_path}")
        # Use method-level parameters if provided, otherwise use class-level defaults
        _max_iter = max_iter if max_iter is not None else self.max_ica_iter
        _max_sources = self.max_sources
        _device = device if device is not None else self.device
        _tolerance = tolerance if tolerance is not None else self.ica_tolerance

        # Store original device if we're changing it temporarily
        original_device = self.device
        if device is not None:
            self.device = self._select_device(device)

        try:
            if method == 'fastICA':
                source, B, spike_train = self._fastICA_(
                    self._preprocessed, _max_sources, _max_iter, tolerance=_tolerance)
            elif method == 'torch':
                source, B, spike_train = self._torch_fastICA_(
                    self._preprocessed, _max_sources, _max_iter, tolerance=_tolerance, device=_device)
            else:
                raise ValueError('method must be either fastICA or torch')
        finally:
            # Restore original device if we changed it
            if device is not None:
                self.device = original_device
        self._raw_source = source
        self._raw_spike_train = spike_train
        self._raw_B = B
        if self.save_ICA:
            self._save_ICA_(self.ICA_path)

    def _save_ICA_(self, path):
        """
        Save the ICA results

        Parameters
        ----------
        path : str
            The path to save the ICA results
        """
        np.savez(path, source=self._raw_source, spike_train=self._raw_spike_train, B=self._raw_B)

    def _load_ICA_(self, path):
        """
        Load the ICA results

        Parameters
        ----------
        path : str
            The path to load the ICA results
        """
        data = np.load(path)
        self._raw_source = data['source']
        self._raw_spike_train = data['spike_train']
        self._raw_B = data['B']

    def remove_motorUnit_duplicates(self, frq=None, min_firing_rate=None, max_firing_rate=None,
                                  max_duplicate_time_diff=None, num_bins=None):
        """
        Remove the duplicate motor units

        Parameters
        ----------
        uncleaned_spkieTrain : numpy.ndarray
            The uncleaned spike train
        uncleaned_source : numpy.ndarray
            The uncleaned sources from the ICA decomposition
        frq : int
            The sampling frequency of the data

        Attributes
        ----------
        spike_train : numpy.ndarray
            The cleaned spike train
        source : numpy.ndarray
            The cleaned sources from the ICA decomposition
        _good_idx : numpy.ndarray
            The cleaned unmixing matrix
        """
        from scipy.spatial.distance import cdist

        spike_train = self._raw_spike_train
        source = self._raw_source

        _frq = frq if frq is not None else self.sampling_frequency
        _min_firing = min_firing_rate if min_firing_rate is not None else self.min_firing_rate
        _max_firing = max_firing_rate if max_firing_rate is not None else self.max_firing_rate
        _max_time_diff = max_duplicate_time_diff if max_duplicate_time_diff is not None \
            else self.max_duplicate_time_diff
        _num_bins = num_bins if num_bins is not None else self.num_bins

        min_firing_interval = 1 / _max_firing  # Minimum time between firings
        time_stamp = np.linspace(1 / _frq, spike_train.shape[0] / _frq, spike_train.shape[0])

        firings = spike_train.sum(axis=0)
        lower_bound_cond = np.where(firings > _min_firing * time_stamp[-1])[0]
        upper_bound_cond = np.where(firings < _max_firing * time_stamp[-1])[0]
        plausible_firings = np.intersect1d(lower_bound_cond, upper_bound_cond)

        for k in plausible_firings:
            spike_time_diff = np.diff(time_stamp[spike_train[:, k] == 1])
            for t in range(len(spike_time_diff)):
                if spike_time_diff[t] < min_firing_interval:
                    if source[t, k] < source[t + 1, k]:
                        spike_train[t, k] = 0
                    else:
                        spike_train[t + 1, k] = 0

        duplicate_sources = []
        for k in plausible_firings:
            if k not in duplicate_sources:
                for j in np.setdiff1d(plausible_firings[plausible_firings != k], duplicate_sources):
                    spike_times_1 = time_stamp[spike_train[:, k] == 1]
                    spike_times_2 = time_stamp[spike_train[:, j] == 1]
                    hist_1, _ = np.histogram(spike_times_1, bins=_num_bins)
                    hist_2, _ = np.histogram(spike_times_2, bins=_num_bins)
                    dist = cdist(hist_1[np.newaxis, :], hist_2[np.newaxis, :], metric='cosine')[0][0]
                    if dist < _max_time_diff:
                        duplicate_sources.append(j)

        good_idx = np.setdiff1d(plausible_firings, duplicate_sources)
        spike_train = spike_train[:, good_idx]
        source = source[:, good_idx]
        self.source = source
        self.spike_train = spike_train
        self._good_idx = good_idx

    def _fast_silhouette(self, data, labels):
        """
        Fast silhouette score calculation for 1D data with 2 clusters.

        Parameters
        ----------
        data : numpy.ndarray
            1D array of values
        labels : numpy.ndarray
            Binary cluster labels (0 or 1)

        Returns
        -------
        float
            Mean silhouette score
        """
        # Split data by cluster
        cluster0 = data[labels == 0]
        cluster1 = data[labels == 1]

        if len(cluster0) == 0 or len(cluster1) == 0:
            return 0.0

        # Calculate distances for each point
        scores = []

        # Process cluster 0
        if len(cluster0) > 1:
            # Vectorized intra-cluster distances for cluster 0
            intra0 = np.abs(cluster0.reshape(-1, 1) - cluster0.reshape(1, -1))
            a0 = np.sum(intra0, axis=1) / (len(cluster0) - 1)  # Exclude self-distance

            # Vectorized inter-cluster distances to cluster 1
            inter0 = np.abs(cluster0.reshape(-1, 1) - cluster1.reshape(1, -1))
            b0 = np.mean(inter0, axis=1)

            # Calculate scores
            s0 = (b0 - a0) / np.maximum(a0, b0)
            scores.extend(s0)

        # Process cluster 1
        if len(cluster1) > 1:
            # Vectorized intra-cluster distances for cluster 1
            intra1 = np.abs(cluster1.reshape(-1, 1) - cluster1.reshape(1, -1))
            a1 = np.sum(intra1, axis=1) / (len(cluster1) - 1)  # Exclude self-distance

            # Vectorized inter-cluster distances to cluster 0
            inter1 = np.abs(cluster1.reshape(-1, 1) - cluster0.reshape(1, -1))
            b1 = np.mean(inter1, axis=1)

            # Calculate scores
            s1 = (b1 - a1) / np.maximum(a1, b1)
            scores.extend(s1)

        return np.mean(scores) if scores else 0.0

    def _compute_score_(self, spike_train, source, max_samples=None):
        """
        Compute the silhouette score of the motor units using optimized sampling

        Parameters
        ----------
        spike_train : numpy.ndarray
            The spike train of the good motor units
        source : numpy.ndarray
            The sources of the good motor units
        frq : int
            The sampling frequency of the data
        max_samples : int
            Maximum number of peaks to use for silhouette calculation

        Returns
        -------
        sil_score : numpy.ndarray
            The silhouette score of the good motor units
        """
        from sklearn.cluster import KMeans
        from scipy.signal import find_peaks

        _max_samples = max_samples if max_samples is not None else self.max_silhouette_samples

        sil_score = np.zeros(spike_train.shape[1])

        for i in range(spike_train.shape[1]):
            pow = np.power(source[:, i], 2)
            loc, _ = find_peaks(pow)
            pks = pow[loc]

            # Sample peaks if there are too many
            if len(pks) > _max_samples:
                sample_idx = np.random.choice(len(pks), _max_samples, replace=False)
                pks = pks[sample_idx]

            kmeans = KMeans(n_clusters=2, n_init=10).fit(pks.reshape(-1, 1))
            idx = kmeans.labels_
            sil_score[i] = self._fast_silhouette(pks.reshape(-1, 1), idx)

        return sil_score

    def compute_score(self, max_silhouette_samples=None):
        """
        Compute the silhouette score of the motor units

        Parameters
        ----------
        max_silhouette_samples : int, optional
            Maximum number of peaks to use for silhouette calculation.
            If None, uses class-level max_silhouette_samples.

        Attributes
        ----------
        sil_score : numpy.ndarray
            The silhouette score of the good motor units
        """
        if self.load_score:
            try:
                self._load_score_(self.score_path)
            except FileNotFoundError:
                print('The silhouette score file does not exist. Computing the silhouette score...')
                self.sil_score = self._compute_score_(
                    self.spike_train, self.source,
                    max_silhouette_samples=max_silhouette_samples)
                if self.save_score:
                    try:
                        self._save_score_(self.score_path)
                    except FileNotFoundError:
                        print('The path to save the silhouette score does not exist.')
        else:
            self.sil_score = self._compute_score_(
                self.spike_train, self.source,
                max_samples=max_silhouette_samples)
            if self.save_score:
                try:
                    self._save_score_(self.score_path)
                except FileNotFoundError:
                    print('The path to save the silhouette score does not exist.')

    def _save_score_(self, path):
        """
        Save the silhouette score of the motor units

        Parameters
        ----------
        path : str
            The path to save the silhouette score
        """
        np.save(path, self.sil_score)

    def _load_score_(self, path):
        """
        Load the silhouette score of the motor units

        Parameters
        ----------
        path : str
            The path to load the silhouette score
        """
        self.sil_score = np.load(path)

    def spikeTrain_plot(
        self, minScore_toPlot=0.93,
        spike_height=0.4, spike_width=0.01,
        color_plot=True, colormap='viridis',
        x_range=None):
        """
        Plot the spike train of the good motor units

        Parameters
        ----------
        minScore_toPlot : float
            The minimum silhouette score of the motor units to be included in the plot
        color_plot : bool
            Whether to use colored plots. Default is True
        spike_height : float
            The relative height of spikes within each MU's band. Default is 0.2.
            Higher values make spikes taller relative to the spacing between MUs.
        spike_width : float
            The width/thickness of each spike line in the plot. Default is 0.01
        colormap : str
            Name of the colormap to use (any matplotlib colormap name). Default is 'viridis'.
            Common options include: 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Spectral', 'RdYlBu', etc.
        x_range : tuple, optional
            Custom x-axis range as (start, end) in seconds. If None, uses full data range.
        """
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt

        frq = self.sampling_frequency
        selected_spikeTrain = self.spike_train[:, self.sil_score > minScore_toPlot]
        order = np.argsort(np.sum(selected_spikeTrain, axis=0))[::-1]
        n_units = selected_spikeTrain.shape[1]

        # Calculate fixed spacing based on number of MUs
        fixed_spacing = self.plot_target_height / self.plot_units_per_height
        # Cap at half target height if fewer than half the target units
        plot_height = int(n_units * fixed_spacing) if (n_units > self.plot_units_per_height / 2) \
            else self.plot_target_height / 2

        # Create a colormap
        if color_plot:
            try:
                colors = self._create_spike_colors_(colormap, plt, n_units)
            except ValueError:
                print(f"Warning: Colormap '{colormap}' not found. Using 'viridis' instead.")
                colors = self._create_spike_colors_('viridis', plt, n_units)

        else:
            colors = ["black"] * n_units

        fig = go.Figure()

        # Fixed unit spacing
        unit_spacing = 1.0

        # Plot each motor unit's spikes
        for r in range(n_units):
            spike_indices = np.where(selected_spikeTrain[:, order[r]] == 1)[0]
            base_y = unit_spacing * r

            # Create vertical lines for each spike
            if len(spike_indices) > 0:
                # Create y coordinates for vertical lines
                y_coords = []
                x_coords = []
                for idx in spike_indices:
                    # Add coordinates for vertical line (bottom to top)
                    x_coords.extend([idx, idx, None])
                    y_coords.extend([base_y - spike_height / 2, base_y + spike_height / 2, None])

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=colors[r],
                        width=spike_width * 100  # Increased width for better visibility
                    ),
                    showlegend=False
                ))

        # Calculate y-axis tick positions for 10% increments
        tick_increment = max(1, n_units // 10)  # At least 1 MU between ticks
        tick_positions = np.arange(0, n_units, tick_increment)
        tick_labels = [str(i) for i in tick_positions]

        fig.update_layout(
            xaxis=dict(
                title="time (sec)",
                range=[0, selected_spikeTrain.shape[0]] if x_range is None else
                      [int(x_range[0] * frq), int(x_range[1] * frq)],
                tickvals=np.arange(0, selected_spikeTrain.shape[0] + 1, selected_spikeTrain.shape[0] // 10),
                ticktext=np.arange(0, int(selected_spikeTrain.shape[0] / frq) + 1,
                                   int(selected_spikeTrain.shape[0] / frq / 10))
            ),
            yaxis=dict(
                title="Motor Unit",
                tickvals=tick_positions,
                ticktext=tick_labels,
                range=[-0.5, n_units + 0.5]
            ),
            height=plot_height,
            showlegend=False,
            plot_bgcolor='white'  # White background
        )

        fig.show()

    def _create_spike_colors_(self, colormap, plt, n_units):
        cmap = plt.get_cmap(colormap)
        # Generate colors from colormap
        colors = []
        for i in range(n_units):
            normalized_idx = i / (n_units - 1) if n_units > 1 else 0
            rgba = cmap(normalized_idx)
            # Convert RGBA to RGB hex, ensuring full opacity
            colors.append(f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})')
        return colors

    def run_decomposition(self, color_plot=True, spike_height=0.2, spike_length=0.01):
        """
        Run the motor-unit decomposition on hdEMG datasets

        Returns
        -------
        None
        """
        # initialize
        # fs = os.sep
        data = self.data
        if isinstance(data, str):
            emg_file = np.load(data)
            data = emg_file['Data'][0]
            self.frq = emg_file['SamplingFrequency']

        max_iter = 200

        # run the decomposition
        extended_emg, _ = self.preprocess(data, self.R, self.whiten_flag, self.SNR)
        if not self.load_ICA:
            uncleaned_source, B, uncleaned_spkieTrain, score = self.run_ICA(extended_emg, self.M, max_iter)
        else:
            warnings.warn("Loading ICA results from a saved file. Change the 'load_ICA' flag if you want to run ICA.")
            ica_results = np.load(f"{self.data}_ica_results.npy")
            uncleaned_source = ica_results['uncleaned_source']
            uncleaned_spkieTrain = ica_results['uncleaned_spkieTrain']
            score = ica_results['score']
            self.B = ica_results['B']
        spike_train, source, good_idx = self.remove_motorUnit_duplicates(uncleaned_spkieTrain, uncleaned_source,
        self.frq)
        silhouette_score = score[good_idx]

        # save the results
        self.motor_unit["spike_train"] = spike_train
        self.motor_unit["source"] = source
        self.motor_unit["good_idx"] = good_idx
        self.motor_unit["silhouette_score"] = silhouette_score
        if self.save_flag:
            np.save(self.output_file, self.motor_unit)

        # plot the motor units in a nice way
        minScore_toPlot = 0.9
        if self.plot_spikeTrain:
            self.spikeTrain_plot(minScore_toPlot, color_plot, spike_height, spike_length)
        return self
