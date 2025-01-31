"""
This module provides visualization functions for EMG signal analysis results.
"""

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


def create_spike_colors(colormap, n_units):
    """
    Generate colors for spike visualization from a matplotlib colormap.

    Parameters
    ----------
    colormap : str
        Name of the matplotlib colormap to use
    n_units : int
        Number of colors to generate

    Returns
    -------
    list
        List of RGB color strings
    """
    cmap = plt.get_cmap(colormap)
    colors = []
    for i in range(n_units):
        normalized_idx = i / (n_units - 1) if n_units > 1 else 0
        rgba = cmap(normalized_idx)
        colors.append(f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})')
    return colors


def plot_spike_train(spike_train, sampling_frequency, silhouette_scores=None, min_score=0.93,
                    spike_height=0.4, spike_width=0.01, color_plot=True, colormap='viridis',
                    x_range=None, target_height=800, units_per_height=40):
    """
    Plot the spike train of motor units.

    Parameters
    ----------
    spike_train : numpy.ndarray
        The spike train data to plot
    sampling_frequency : int
        Sampling frequency of the data in Hz
    silhouette_scores : numpy.ndarray, optional
        Silhouette scores for each motor unit
    min_score : float, optional
        Minimum silhouette score for units to include in plot
    spike_height : float, optional
        The relative height of spikes within each MU's band
    spike_width : float, optional
        The width/thickness of each spike line
    color_plot : bool, optional
        Whether to use colored plots
    colormap : str, optional
        Name of the matplotlib colormap to use
    x_range : tuple, optional
        Custom x-axis range as (start, end) in seconds
    target_height : int, optional
        Target plot height in pixels
    units_per_height : int, optional
        Number of units per height unit

    Returns
    -------
    None
    """
    if silhouette_scores is not None:
        selected_spikeTrain = spike_train[:, silhouette_scores > min_score]
    else:
        selected_spikeTrain = spike_train

    order = np.argsort(np.sum(selected_spikeTrain, axis=0))[::-1]
    n_units = selected_spikeTrain.shape[1]

    # Calculate fixed spacing based on number of MUs
    fixed_spacing = target_height / units_per_height
    # Cap at half target height if fewer than half the target units
    plot_height = int(n_units * fixed_spacing) if (n_units > units_per_height / 2) \
        else target_height / 2

    # Create a colormap
    if color_plot:
        try:
            colors = create_spike_colors(colormap, n_units)
        except ValueError:
            print(f"Warning: Colormap '{colormap}' not found. Using 'viridis' instead.")
            colors = create_spike_colors('viridis', n_units)
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
                  [int(x_range[0] * sampling_frequency), int(x_range[1] * sampling_frequency)],
            tickvals=np.arange(0, selected_spikeTrain.shape[0] + 1, selected_spikeTrain.shape[0] // 10),
            ticktext=np.arange(0, int(selected_spikeTrain.shape[0] / sampling_frequency) + 1,
                               int(selected_spikeTrain.shape[0] / sampling_frequency / 10))
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
