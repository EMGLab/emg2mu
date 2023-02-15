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
            sigPower = np.sum(np.abs(sig)**2)/sig.size
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
            sigPower = 10**(sigPower/10)
        reqSNR = 10**(reqSNR/10)
    
    # Check for invalid signal power and SNR for linear scale
    if isLinearScale and sigPower <= 0:
        raise ValueError("The signal power must be positive for linear scale.")
    if isLinearScale and reqSNR <= 0:
        raise ValueError("The SNR must be positive for linear scale.")
    
    noisePower = sigPower/reqSNR
    
    # Add noise
    if np.iscomplexobj(sig):
        noise = np.sqrt(noisePower/2) * (np.random.randn(*sig.shape) + 1j*np.random.randn(*sig.shape))
    else:
        noise = np.sqrt(noisePower) * np.random.randn(*sig.shape)
    
    y = sig + noise
    return y


class EMG:
        """
    motor-unit decomposition on hdEMG datasets
    
    This function and the helper files are mainly a python implementation of the code accompanied with
    Hyser Dataset by Jian et. al.
    The original code and the dataset is also available at PhysioNet

    Parameters:
    data : str or numpy.ndarray
        The path to the hdEMG data. If 'data' is pointing to the location of the data file, the data file must be a MAT array.
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
        Adding white noise to the EMG mixutre. Uses Communication Toolbox AWGN function. Default = Inf for not injecting any artificial noise.
    silhouette_threshold : float, optional
        The silhouette threshold to detect the good motor units. Default = 0.6
    output_file : str, optional
        The path that the files should be saved there. The function does not create the path, rather uses it. Default is the is the 'sample' path of the toolbox.
    save_flag : int, optional
        Whether the files are saved or not, default is 0, so it is NOT saving your output.
    plot_spikeTrain : int, optional
        Whether to plot the resulting spike trains. Default is 1
    load_ICA : int, optional
        Whether to load precomputed ICA results for debugging. Default is 0
    
    Returns:
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
                 save_flag=0, plot_spikeTrain=1, load_ICA=0):
        self.data = data
        self.data_mode = data_mode
        self.sampling_frequency = sampling_frequency
        self.extension_parameter = extension_parameter
        self.max_sources = max_sources
        self.whiten_flag = whiten_flag
        self.inject_noise = inject_noise
        self.silhouette_threshold = silhouette_threshold
        self.output_file = output_file
        self.save_flag = save_flag
        self.plot_spikeTrain = plot_spikeTrain
        self.load_ICA = load_ICA
        self.motor_unit = {}


    def preprocess_emg(self, data_mode='monopolar', whiten_flag=True, R=4, SNR=np.inf, array_shape=[8,8]):
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
            Adding white noise to the EMG mixture. Uses Communication Toolbox AWGN function. Default = Inf for not injecting any artificial noise.
        R : int
            The number of times to repeat the data blocks, see the Hyser paper for more detail. Default = 4
        array_shape : list-like
            The first element will be used to calculate the bipolar activity if the bipolar flag is on for the 'data_mode'.
        
        Returns
        -------
        preprocessed_data : array-like
            The preprocessed EMG data
        W : array-like
            The whitening matrix used for preprocessing
        '''
        # data can come in column or row format, but needs to become the column format where the
        emg = self.emg
        num_chan = min(emg.shape)  # Let's assume that we have more than 64 frames
        if num_chan != emg.shape[1]:
            emg = emg.T

        # Add white noise
        if not np.isinf(SNR):
            emg = self.awgn(emg, SNR, 'dB')
        
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
                extended_emg[i+1:, emg.shape[1] * i : emg.shape[1] * (i + 1)] = emg[:-i, :]
        
        if whiten_flag:
            whitened_data, _, _, W = self.whiten(extended_emg)
            preprocessed_emg = whitened_data
        else:
            preprocessed_emg = extended_emg

        self._preprocessed_emg = preprocessed_emg
        return self
    
    def _torch_fastICA(emg, M, max_iter):
        """
        Run the ICA decomposition
        
        Parameters:
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
        device = torch.device("mps" if torch.has_mps() else "gpu")
        if not device:
            device = torch.device("cpu")
        tolerance = 10e-4
        emg = emg.T
        num_chan, frames = emg.shape
        B = torch.zeros(num_chan, M, device=device)
        spike_train = torch.zeros(frames, M, device=device)
        source = torch.zeros(frames, M, device=device)
        score = torch.zeros(1, M, device=device)
        print(f'running ICA for {M} sources')
        for i in range(M):
            w = torch.empty(num_chan, 2, device=device).normal_(mean=0, std=1)
            for n in range(2, max_iter):
                dot_product = torch.matmul(w[:, n-1].unsqueeze(1), emg)
                A = 2 * torch.mean(dot_product)
                w_new = emg * dot_product.pow(2) - A * w[:, n-1].unsqueeze(1)
                w_new = w_new - torch.matmul(torch.matmul(B[:, i].unsqueeze(0), B[:, i].unsqueeze(1)), w_new)
                w_new = w_new / w_new.norm()
                if (w[:, n-1].unsqueeze(1).matmul(w_new) - 1).abs() <= tolerance:
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

    def _fastICA(emg, M, max_iter):
        tolerance = 10e-4
        num_chan, frames = emg.shape
        B = np.zeros((num_chan, 1))
        spike_train = np.zeros((frames, M))
        source = np.zeros((frames, M))
        score = np.zeros(1, M)
        print(f"running ICA for {M} sources")
        for i in range(M):
            w = []
            w.append(np.random.randn(num_chan, 1))
            w.append(np.random.randn(num_chan, 1))
            for n in range(2, max_iter):
                if abs(np.dot(w[n].T, w[n-1]) - 1) > tolerance:
                    A = np.mean(2 * np.dot(w[n].T, emg))
                    w.append(emg @ ((np.dot(w[n].T, emg).T) ** 2) - A * w[n])
                    w[-1] = w[-1] - np.dot(np.dot(B, B.T), w[-1])
                    w[-1] = w[-1] / np.linalg.norm(w[-1])
                else:
                    break
            source[:, i] = np.dot(w[-1].T, emg)
            pks = np.power(source[:, i], 2)
            loc = np.array([index for index, value in enumerate(pks) if value == np.max(pks)])
            idx = np.array([1 if i < len(pks) / 2 else 2 for i in range(len(pks))])
            sil_score = np.array([(pks[i] - np.mean([pks[j] for j in range(len(pks)) if idx[j] == idx[i]])) / np.std([pks[j] for j in range(len(pks)) if idx[j] == idx[i]]) for i in range(len(pks))])
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
    
    # def remove_motorUnit_duplicates(self, uncleaned_spkieTrain, uncleaned_source, frq):
    #     """
    #     Remove the duplicate motor units
        
    #     Parameters:
    #     uncleaned_spkieTrain : numpy.ndarray
    #         The uncleaned spike train
    #     uncleaned_source : numpy.ndarray
    #         The uncleaned sources from the ICA decomposition
    #     frq : int
    #         The sampling frequency of the data
        
    #     Returns:
    #     spike_train : numpy.ndarray
    #         The spike train of the good motor units
    #     source : numpy.ndarray
    #         The sources of the good motor units
    #     good_idx : numpy.ndarray
    #         The indices of the good motor units
    #     """
    #     pass
    
    # def spikeTrain_plot(self, spike_train, frq, silhouette_score, minScore_toPlot):
    #     """
    #     Plot the spike train of the good motor units
        
    #     Parameters:
    #     spike_train : numpy.ndarray
    #         The spike train of the good motor units
    #     frq : int
    #         The sampling frequency of the data
    #     silhouette_score : numpy.ndarray
    #         The silhouette score of the good motor units
    #     minScore_toPlot : float
    #         The minimum silhouette score of the motor units to be included in the plot
    #     """
    #     pass
    
    def run_decomposition(self):
        """
        Run the motor-unit decomposition on hdEMG datasets
        
        Returns:
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
        extended_emg, _ = self.preprocess_emg(data, self.R, self.whiten_flag, self.SNR)
        if not self.load_ICA:
            uncleaned_source, B, uncleaned_spkieTrain, score = self.run_ICA(extended_emg, self.M, max_iter)
        else:
            warnings.warn("Loading ICA results from a saved file. Change the 'load_ICA' flag if you want to run ICA.")
            ica_results = np.load(f"{self.data}_ica_results.npy")
            uncleaned_source = ica_results['uncleaned_source']
            uncleaned_spkieTrain = ica_results['uncleaned_spkieTrain']
            score = ica_results['score']
            B = ica_results['B']
        spike_train, source, good_idx = self.remove_motorUnit_duplicates(uncleaned_spkieTrain, uncleaned_source, self.frq)
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