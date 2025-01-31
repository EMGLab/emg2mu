"""
This module provides functions for motor unit detection and duplicate removal in EMG analysis.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


def fast_silhouette(data, labels):
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


def compute_silhouette_scores(spike_train, source, max_samples=1000):
    """
    Compute silhouette scores for motor units using optimized sampling.

    Parameters
    ----------
    spike_train : numpy.ndarray
        The spike train data
    source : numpy.ndarray
        The source signals from ICA decomposition
    max_samples : int
        Maximum number of peaks to use for silhouette calculation

    Returns
    -------
    numpy.ndarray
        Array of silhouette scores for each motor unit
    """
    sil_score = np.zeros(spike_train.shape[1])

    for i in range(spike_train.shape[1]):
        pow = np.power(source[:, i], 2)
        loc, _ = find_peaks(pow)
        pks = pow[loc]

        # Sample peaks if there are too many
        if len(pks) > max_samples:
            sample_idx = np.random.choice(len(pks), max_samples, replace=False)
            pks = pks[sample_idx]

        kmeans = KMeans(n_clusters=2, n_init=10).fit(pks.reshape(-1, 1))
        idx = kmeans.labels_
        sil_score[i] = fast_silhouette(pks.reshape(-1, 1), idx)

    return sil_score


def remove_duplicates(spike_train, source, sampling_frequency, min_firing_rate=4, max_firing_rate=35,
                     max_duplicate_time_diff=0.01, num_bins=100):
    """
    Remove duplicate motor units from decomposition results.

    Parameters
    ----------
    spike_train : numpy.ndarray
        The uncleaned spike train
    source : numpy.ndarray
        The uncleaned sources from ICA decomposition
    sampling_frequency : int
        The sampling frequency of the data
    min_firing_rate : float
        Minimum firing rate in Hz for valid motor units
    max_firing_rate : float
        Maximum firing rate in Hz for valid motor units
    max_duplicate_time_diff : float
        Maximum time difference for considering motor units as duplicates
    num_bins : int
        Number of bins used for histogram analysis in duplicate detection

    Returns
    -------
    tuple
        (cleaned_spike_train, cleaned_source, good_indices)
    """
    min_firing_interval = 1 / max_firing_rate  # Minimum time between firings
    time_stamp = np.linspace(1 / sampling_frequency, spike_train.shape[0] / sampling_frequency,
                            spike_train.shape[0])

    firings = spike_train.sum(axis=0)
    lower_bound_cond = np.where(firings > min_firing_rate * time_stamp[-1])[0]
    upper_bound_cond = np.where(firings < max_firing_rate * time_stamp[-1])[0]
    plausible_firings = np.intersect1d(lower_bound_cond, upper_bound_cond)

    # Remove spikes that are too close together
    for k in plausible_firings:
        spike_time_diff = np.diff(time_stamp[spike_train[:, k] == 1])
        for t in range(len(spike_time_diff)):
            if spike_time_diff[t] < min_firing_interval:
                if source[t, k] < source[t + 1, k]:
                    spike_train[t, k] = 0
                else:
                    spike_train[t + 1, k] = 0

    # Find duplicate sources
    duplicate_sources = []
    for k in plausible_firings:
        if k not in duplicate_sources:
            for j in np.setdiff1d(plausible_firings[plausible_firings != k], duplicate_sources):
                spike_times_1 = time_stamp[spike_train[:, k] == 1]
                spike_times_2 = time_stamp[spike_train[:, j] == 1]
                hist_1, _ = np.histogram(spike_times_1, bins=num_bins)
                hist_2, _ = np.histogram(spike_times_2, bins=num_bins)
                dist = cdist(hist_1[np.newaxis, :], hist_2[np.newaxis, :], metric='cosine')[0][0]
                if dist < max_duplicate_time_diff:
                    duplicate_sources.append(j)

    good_idx = np.setdiff1d(plausible_firings, duplicate_sources)
    return spike_train[:, good_idx], source[:, good_idx], good_idx
