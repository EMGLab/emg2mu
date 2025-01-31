"""
This module provides ICA (Independent Component Analysis) implementations for EMG signal processing.
Includes both standard CPU-based and PyTorch-accelerated implementations.
"""

from tqdm import tqdm
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F


def fastICA(extended_emg, M, max_iter, tolerance=1e-5):
    """
    Run the ICA decomposition using standard CPU implementation.

    Parameters
    ----------
    extended_emg : numpy.ndarray
        The preprocessed extended EMG data
    M : int
        Maximum number of sources being decomposed by (FAST) ICA
    max_iter : int
        Maximum iterations for the (FAST) ICA decomposition
    tolerance : float
        Convergence tolerance for ICA

    Returns
    -------
    source : numpy.ndarray
        The uncleaned sources from the ICA decomposition
    B : numpy.ndarray
        The unmixing matrix
    spike_train : numpy.ndarray
        The uncleaned spike train
    """
    emg = extended_emg.T
    num_chan, frames = emg.shape
    B = np.zeros((num_chan, M))
    spike_train = np.zeros((frames, M))
    source = np.zeros((frames, M))
    print(f"Running ICA for {M} sources...")
    
    pbar = tqdm(range(M), desc="Processing sources", unit="source")
    for i in pbar:
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
        kmeans = KMeans(n_clusters=2, n_init=10).fit(pks.reshape(-1, 1))
        idx = kmeans.labels_
        
        if sum(idx == 0) <= sum(idx == 1):
            spike_loc = loc[idx == 0]
        else:
            spike_loc = loc[idx == 1]
            
        spike_train[spike_loc, i] = 1
        B[:, i] = w[-1].flatten()
        pbar.set_postfix({"source": f"{i+1}/{M}"})
            
    print("ICA decomposition completed")
    return source, B, spike_train


def torch_fastICA(extended_emg, M, max_iter, tolerance=1e-5, device='cuda'):
    """
    Run the ICA decomposition using PyTorch for GPU acceleration.

    Parameters
    ----------
    extended_emg : numpy.ndarray
        The preprocessed extended EMG data
    M : int
        Maximum number of sources being decomposed by (FAST) ICA
    max_iter : int
        Maximum iterations for the (FAST) ICA decomposition
    tolerance : float
        Convergence tolerance for ICA
    device : str
        PyTorch device to use ('cuda', 'mps', or 'cpu')

    Returns
    -------
    source : numpy.ndarray
        The uncleaned sources from the ICA decomposition
    B : numpy.ndarray
        The unmixing matrix
    spike_train : numpy.ndarray
        The uncleaned spike train
    """
    emg = torch.tensor(extended_emg.T, dtype=torch.float32, device=device)
    num_chan, frames = emg.shape
    B = torch.zeros((num_chan, M), dtype=torch.float32, device=device)
    spike_train = torch.zeros((frames, M), dtype=torch.float32, device=device)
    source = torch.zeros((frames, M), dtype=torch.float32, device=device)
    
    print(f"Running ICA for {M} sources...")
    pbar = tqdm(range(M), desc="Processing sources", unit="source")
    for i in pbar:
        w = []
        w.append(torch.randn(num_chan, 1, device=device, dtype=torch.float32))
        w.append(torch.randn(num_chan, 1, device=device, dtype=torch.float32))
        
        for n in range(1, max_iter):
            if abs(torch.matmul(w[n].T, w[n - 1]) - 1) > tolerance:
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
        
        if sum(idx == 1) <= sum(idx == 0):
            spike_loc = loc[idx == 1]
        else:
            spike_loc = loc[idx == 0]
            
        spike_train[spike_loc, i] = 1
        B[:, i] = w[-1].flatten()
        pbar.set_postfix({"source": f"{i+1}/{M}"})
            
    print("ICA decomposition completed")
    return source.cpu().numpy(), B.cpu().numpy(), spike_train.cpu().numpy()


def select_device(device_preference='auto'):
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
