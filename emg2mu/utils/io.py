"""
This module provides I/O utility functions for EMG data handling.
"""

import numpy as np
import scipy.io as sio


def load_mat_data(file_path):
    """
    Load EMG data from a MAT file.

    Parameters
    ----------
    file_path : str
        Path to the MAT file

    Returns
    -------
    tuple
        (data, sampling_frequency)
        - data: The EMG data array
        - sampling_frequency: The sampling frequency in Hz

    Raises
    ------
    ValueError
        If the file format is invalid or required fields are missing
    """
    try:
        emg_file = sio.loadmat(file_path)
        data = emg_file['Data'][0, 0]
        sampling_frequency = emg_file['SamplingFrequency'].item()
        return data, sampling_frequency
    except Exception as e:
        raise ValueError(f"Error loading MAT file: {str(e)}")


def save_results(file_path, spike_train, source, good_idx, silhouette_score=None):
    """
    Save decomposition results to a NPZ file.

    Parameters
    ----------
    file_path : str
        Path to save the results
    spike_train : numpy.ndarray
        The spike train data
    source : numpy.ndarray
        The source signals
    good_idx : numpy.ndarray
        Indices of valid motor units
    silhouette_score : numpy.ndarray, optional
        Silhouette scores for the motor units

    Returns
    -------
    None
    """
    save_dict = {
        'spike_train': spike_train,
        'source': source,
        'good_idx': good_idx
    }
    if silhouette_score is not None:
        save_dict['silhouette_score'] = silhouette_score
    
    np.savez(file_path, **save_dict)


def load_results(file_path):
    """
    Load decomposition results from a NPZ file.

    Parameters
    ----------
    file_path : str
        Path to the results file

    Returns
    -------
    dict
        Dictionary containing the loaded results with keys:
        - spike_train
        - source
        - good_idx
        - silhouette_score (if available)

    Raises
    ------
    ValueError
        If the file format is invalid or required fields are missing
    """
    try:
        data = np.load(file_path)
        results = {}
        for key in ['spike_train', 'source', 'good_idx']:
            if key not in data:
                raise ValueError(f"Required field '{key}' not found in results file")
            results[key] = data[key]
        
        # Optional silhouette score
        if 'silhouette_score' in data:
            results['silhouette_score'] = data['silhouette_score']
        
        return results
    except Exception as e:
        raise ValueError(f"Error loading results file: {str(e)}")


def save_ica_results(file_path, source, spike_train, B):
    """
    Save ICA decomposition results to a NPZ file.

    Parameters
    ----------
    file_path : str
        Path to save the ICA results
    source : numpy.ndarray
        The source signals from ICA
    spike_train : numpy.ndarray
        The spike train data
    B : numpy.ndarray
        The unmixing matrix

    Returns
    -------
    None
    """
    np.savez(file_path, source=source, spike_train=spike_train, B=B)


def load_ica_results(file_path):
    """
    Load ICA decomposition results from a NPZ file.

    Parameters
    ----------
    file_path : str
        Path to the ICA results file

    Returns
    -------
    tuple
        (source, spike_train, B)
        - source: The source signals from ICA
        - spike_train: The spike train data
        - B: The unmixing matrix

    Raises
    ------
    ValueError
        If the file format is invalid or required fields are missing
    """
    try:
        data = np.load(file_path)
        if not all(key in data for key in ['source', 'spike_train', 'B']):
            raise ValueError("Missing required fields in ICA results file")
        return data['source'], data['spike_train'], data['B']
    except Exception as e:
        raise ValueError(f"Error loading ICA results file: {str(e)}")


def save_silhouette_scores(file_path, scores):
    """
    Save silhouette scores to a NPY file.

    Parameters
    ----------
    file_path : str
        Path to save the silhouette scores
    scores : numpy.ndarray
        Array of silhouette scores

    Returns
    -------
    None
    """
    np.save(file_path, scores)


def load_silhouette_scores(file_path):
    """
    Load silhouette scores from a NPY file.

    Parameters
    ----------
    file_path : str
        Path to the silhouette scores file

    Returns
    -------
    numpy.ndarray
        Array of silhouette scores

    Raises
    ------
    ValueError
        If the file cannot be loaded
    """
    try:
        return np.load(file_path)
    except Exception as e:
        raise ValueError(f"Error loading silhouette scores file: {str(e)}")
