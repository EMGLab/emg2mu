"""
This module provides preprocessing functions for EMG signal processing.

Functions:
    - awgn: Add white Gaussian noise to a signal
    - whiten: Whiten a matrix using various methods
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
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix with data examples along the first dimension
    method : str
        Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
        'pca_cor', or 'cholesky'.
        
    Returns
    -------
    X_hat : numpy.ndarray
        Whitened data matrix

    References
    ----------
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
