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
        Sampling frequency of the data, default = 2048 Hz
    extension_parameter : int, optional
        The number of times to repeat the data blocks, see the Hyser paper for more detail. Default = 4
    max_sources : int, optional
        Maximum iterations for the (FAST) ICA decompsition. Default = 300
    whiten_flag : int, optional
        Whether to whiten the data prior to the ICA. Default = 1
    inject_noise : float, optional
        Adding white noise to the EMG mixutre. Uses the equivalent of the Communication Toolbox AWGN function.
        Default = Inf for not injecting any artificial noise.
    silhouette_threshold : float, optional
        The silhouette threshold to detect the good motor units. Default = 0.6
    output_file : str, optional
        The path that the files should be saved there. The function does not create the path, rather uses it.
        Default is the is the 'sample' path of the toolbox.
    save_flag : int, optional
        Whether the files are saved or not, default is 0, so it is NOT saving your output.
    plot_spikeTrain : int, optional
        Whether to plot the resulting spike trains. Default is 1
    load_ICA : int, optional
        Whether to load precomputed ICA results for debugging. Default is 0

    Returns
    -------
    motor_unit : dict
        The structure including the following fields:
        spike_train
        waveform
        ica_weight
        whiten_matrix
        silhouette
    """

    def __init__(self, data, data_mode='monopolar', sampling_frequency=2048, extension_parameter=4, max_sources=300,
                 whiten_flag=1, inject_noise=np.inf, silhouette_threshold=0.6, output_file='sample_decomposed',
                 max_ica_iter=100, save_flag=0, plot_spikeTrain=1, load_ICA=0):
        if isinstance(data, str):
            try:
                import scipy.io as sio
                emg_file = sio.loadmat(data)
                self.data = emg_file['Data'][0, 0]
                self.sampling_frequency = emg_file['SamplingFrequency']
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
        self.save_flag = save_flag
        self.plot_spikeTrain = plot_spikeTrain
        self.load_ICA = load_ICA
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

    def _torch_fastICA(self, M, max_iter):
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
                Returns:
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
        device = torch.device("mps" if torch.has_mps else "gpu")
        if not device:
            device = torch.device("cpu")
        tolerance = 10e-5
        emg = self._preprocessed.T
        emg = torch.from_numpy(np.float32(emg)).to(device)
        num_chan, frames = emg.shape
        B = torch.zeros(num_chan, M, device=device)
        spike_train = torch.zeros(frames, M, device=device)
        source = torch.zeros(frames, M, device=device)
        score = torch.zeros(1, M, device=device)
        print(f'running ICA for {M} sources')
        for i in range(M):
            w = torch.empty(num_chan, 2, device=device).normal_(mean=0, std=1)
            for n in range(1, max_iter):
                dot_product = torch.matmul(w[:, n - 1].unsqueeze(1).T, emg)
                A = 2 * torch.mean(dot_product)
                w_new = emg * dot_product.pow(2) - A * w[:, n - 1].unsqueeze(1)
                w_new = w_new - torch.matmul(torch.matmul(B[:, i].unsqueeze(0), B[:, i].unsqueeze(1)), w_new)
                w_new = w_new / w_new.norm()
                if (w[:, n - 1].unsqueeze(1).matmul(w_new) - 1).abs() <= tolerance:
                    break
                w = torch.cat((w, w_new), dim=1)
            source[:, i] = torch.matmul(w[:, -1].unsqueeze(1), emg).squeeze()
            pks, loc = torch.max(source[:, i].pow(2), dim=0)
            idx, _ = torch.kmeans(pks, 2, max_iter=1)
            sil_score = np.silhouette_samples(pks.numpy(), idx.numpy())
            score[0, i] = (sil_score[idx.numpy() == 0].mean() + sil_score[idx.numpy() == 1].mean()) / 2
            if (idx == 0).sum() <= (idx == 1).sum():
                spike_loc = loc[idx == 0]
            else:
                spike_loc = loc[idx == 1]
            spike_train[spike_loc, i] = 1
            B[:, i] = w[:, -1]
            print('.', end='')
        spike_train = spike_train.to_sparse()
        print('\nICA decomposition is completed')
        return source, B, spike_train, score

    def _fastICA(self, M, max_iter):
        from scipy.signal import find_peaks
        tolerance = 10e-5
        emg = self._preprocessed.T
        num_chan, frames = emg.shape
        B = np.zeros((num_chan, M))
        spike_train = np.zeros((frames, M))
        source = np.zeros((frames, M))
        score = np.zeros(M)
        print(f"running ICA for {M} sources")
        for i in range(M):
            # print(f"running ICA for source number {i + 1}")
            w = []
            w.append(np.random.randn(num_chan, 1))
            w.append(np.random.randn(num_chan, 1))
            for n in range(1, max_iter):
                if abs(np.dot(w[n].T, w[n - 1]) - 1) > tolerance:
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
            idx = np.array([1 if i < len(pks) / 2 else 2 for i in range(len(pks))])
            mean = np.zeros_like(pks)
            std = np.zeros_like(pks)
            for i, idx_val in enumerate([1, 2]):
                idx_mask = (idx == idx_val)
                pks_masked = np.where(idx_mask, pks, np.nan)
                mean[idx_mask] = np.nanmean(pks_masked)
                std[idx_mask] = np.nanstd(pks_masked)
            sil_score = (pks - mean) / std
            score[i] = (np.mean(sil_score[idx == 1]) + np.mean(sil_score[idx == 2])) / 2
            if sum(idx == 1) <= sum(idx == 2):
                spike_loc = loc[idx == 1]
            else:
                spike_loc = loc[idx == 2]
            spike_train[spike_loc, i] = 1
            B[:, i] = w[-1].flatten()
            print(".", end="")
        spike_train = np.array(spike_train)
        print("\nICA decomposition is completed")
        return source, B, spike_train, score

    def run_ICA(self, method='fastICA'):
        max_iter = self.max_ica_iter
        max_sources = self.max_sources
        if method == 'fastICA':
            source, B, spike_train, score = self._fastICA(max_sources, max_iter)
        elif method == 'torch':
            source, B, spike_train, score = self._torch_fastICA(max_sources, max_iter)
        else:
            raise ValueError('method must be either fastICA or torch')
        return source, B, spike_train, score

    def remove_motorUnit_duplicates(self, spike_train, source, frq=2048):
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

        Returns
        -------
        spike_train : numpy.ndarray
            The spike train of the good motor units
        source : numpy.ndarray
            The sources of the good motor units
        good_idx : numpy.ndarray
            The indices of the good motor units
        """
        from scipy.spatial.distance import cdist

        min_firing = 4
        max_firing = 35
        min_firing_interval = 1 / max_firing
        time_stamp = np.linspace(1 / frq, spike_train.shape[0] / frq, spike_train.shape[0])

        firings = spike_train.sum(axis=0)
        lower_bound_cond = np.where(firings > min_firing * time_stamp[-1])[0]
        upper_bound_cond = np.where(firings < max_firing * time_stamp[-1])[0]
        plausible_firings = np.intersect1d(lower_bound_cond, upper_bound_cond)

        for k in plausible_firings:
            spike_time_diff = np.diff(time_stamp[spike_train[:, k] == 1])
            for t in range(len(spike_time_diff)):
                if spike_time_diff[t] < min_firing_interval:
                    if source[t, k] < source[t + 1, k]:
                        spike_train[t, k] = 0
                    else:
                        spike_train[t + 1, k] = 0

        max_time_diff = 0.01
        num_bins = 10
        duplicate_sources = []
        for k in plausible_firings:
            if k not in duplicate_sources:
                for j in np.setdiff1d(plausible_firings[plausible_firings != k], duplicate_sources):
                    spike_times_1 = time_stamp[spike_train[:, k] == 1]
                    spike_times_2 = time_stamp[spike_train[:, j] == 1]
                    hist_1, _ = np.histogram(spike_times_1, bins=num_bins)
                    hist_2, _ = np.histogram(spike_times_2, bins=num_bins)
                    dist = cdist(hist_1[np.newaxis, :], hist_2[np.newaxis, :], metric='cosine')[0][0]
                    if dist < max_time_diff:
                        duplicate_sources.append(j)

        good_idx = np.setdiff1d(plausible_firings, duplicate_sources)
        spike_train = spike_train[:, good_idx]
        source = source[:, good_idx]
        return spike_train, source, good_idx

    def spikeTrain_plot(self, spike_train, frq, sil_score, minScore_toPlot=0.7):
        """
        Plot the spike train of the good motor units

        Parameters
        ----------
        spike_train : numpy.ndarray
            The spike train of the good motor units
        frq : int
            The sampling frequency of the data
        silhouette_score : numpy.ndarray
            The silhouette score of the good motor units
        minScore_toPlot : float
            The minimum silhouette score of the motor units to be included in the plot
        """
        import plotly.graph_objects as go
        # import pandas as pd

        selected_spikeTrain = spike_train[:, sil_score > minScore_toPlot]
        order = np.argsort(np.sum(selected_spikeTrain, axis=1))[::-1]
        bar_height = 0.2
        fig = go.Figure()
        for r in range(selected_spikeTrain.shape[1]):
            signRugVal = np.abs(np.sign(selected_spikeTrain[:, order[r]]))
            rug_x = np.where(signRugVal == 1)[0]
            rug_y = [bar_height * r] * len(rug_x)
            fig.add_scatter(x=rug_x, y=rug_y, mode="markers", marker=dict(size=2, color="black"))
        fig.update_layout(
            xaxis=dict(title="time (sec)", tickvals=np.linspace(0, selected_spikeTrain.shape[0], 10),
                       ticktext=np.round(np.linspace(0, selected_spikeTrain.shape[0] / frq, 10))),
            yaxis=dict(title="Motor Unit", range=[0, selected_spikeTrain.shape[1] * bar_height]), height=400)
        fig.show()

    def run_decomposition(self):
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
            self.plot_spikeTrain(spike_train, self.frq, silhouette_score, minScore_toPlot)
        return self
