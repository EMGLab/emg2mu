"""
This module provides visualization functions for EMG signal analysis results.
"""

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from plotly.subplots import make_subplots


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
    if n_units <= 0:
        return []
    elif n_units == 1:
        rgba = cmap(0)
        return [f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})']
    else:
        for i in range(n_units):
            normalized_idx = i / (n_units - 1)
            rgba = cmap(normalized_idx)
            colors.append(f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})')
        return colors


def plot_waveforms(source, spike_train, sampling_frequency, silhouette_scores=None,
                  min_score=0.93, **kwargs):
    """
    Plot average waveforms for each motor unit with optional individual spikes and confidence intervals.

    Parameters
    ----------
    source : numpy.ndarray
        The ICA source signals matrix
    spike_train : numpy.ndarray
        The spike train data matrix
    sampling_frequency : int
        Sampling frequency of the data in Hz
    window_size : float, optional
        Time window around each spike in seconds (default = 0.005, i.e., ±5ms)
    plot_individual : bool, optional
        Whether to plot individual spikes in background (default = False)
    confidence_interval : bool, optional
        Whether to show confidence intervals (default = True)
    alpha : float, optional
        Transparency for individual spikes (default = 0.1)
    colormap : str, optional
        Name of the matplotlib colormap to use (default = 'viridis')
    silhouette_scores : numpy.ndarray, optional
        Silhouette scores for each motor unit
    min_score : float, optional
        Minimum silhouette score for units to include (default = 0.93)

    Returns
    -------
    None
    """
    # Filter motor units based on silhouette scores if provided
    if silhouette_scores is not None:
        selected_spike_train = spike_train[:, silhouette_scores > min_score]
        selected_source = source[:, silhouette_scores > min_score]
    else:
        selected_spike_train = spike_train
        selected_source = source

    n_units = selected_spike_train.shape[1]
    if n_units == 0:
        raise ValueError("No motor units meet the silhouette score threshold")

    # Set default parameters
    window_size = kwargs.get('window_size', 0.005)
    plot_individual = kwargs.get('plot_individual', False)
    confidence_interval = kwargs.get('confidence_interval', True)
    alpha = kwargs.get('alpha', 0.1)
    colormap = kwargs.get('colormap', 'viridis')
    n_cols = kwargs.get('n_cols', 5)
    subplot_height = kwargs.get('subplot_height', 200)
    subplot_width = kwargs.get('subplot_width', 300)

    # Calculate window size in samples
    window_samples = int(window_size * sampling_frequency)
    time_vector = np.linspace(-window_size, window_size, 2 * window_samples + 1) * 1000  # Convert to ms

    # Create color map
    colors = create_spike_colors(colormap, n_units)


    # Calculate grid dimensions
    n_rows = (n_units + n_cols - 1) // n_cols  # Ceiling division

    # Create subplot titles with scores
    subplot_titles = []
    for i in range(n_units):
        score_text = f" (score: {silhouette_scores[i]:.2f})" if silhouette_scores is not None else ""
        subplot_titles.append(f'Motor Unit {i+1}<span style="font-size:10px">{score_text}</span>')

    # Create subplot grid
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    # Process each motor unit
    for i in range(n_units):
        # Find spike timestamps
        spike_indices = np.where(selected_spike_train[:, i])[0]

        if len(spike_indices) == 0:
            continue

        # Extract waveform segments
        waveforms = []
        for idx in spike_indices:
            if idx - window_samples >= 0 and idx + window_samples + 1 <= len(selected_source):
                segment = selected_source[idx - window_samples:idx + window_samples + 1, i]
                waveforms.append(segment)

        if not waveforms:
            continue

        waveforms = np.array(waveforms)

        # Calculate mean waveform and confidence intervals
        mean_waveform = np.mean(waveforms, axis=0)
        if confidence_interval:
            ci = 1.96 * np.std(waveforms, axis=0) / np.sqrt(len(waveforms))
            upper_ci = mean_waveform + ci
            lower_ci = mean_waveform - ci

        # Plot individual spikes if requested
        if plot_individual:
            for waveform in waveforms:
                fig.add_trace(
                    go.Scatter(x=time_vector, y=waveform,
                             mode='lines',
                             line=dict(color=colors[i], width=1),
                             opacity=alpha,
                             showlegend=False),
                    row=(i // n_cols) + 1,
                    col=(i % n_cols) + 1
                )

        # Plot confidence intervals if requested
        if confidence_interval:
            fig.add_trace(
                go.Scatter(x=time_vector, y=upper_ci,
                          mode='lines',
                          line=dict(width=0),
                          showlegend=False),
                row=(i // n_cols) + 1,
                col=(i % n_cols) + 1
            )
            fig.add_trace(
                go.Scatter(x=time_vector, y=lower_ci,
                          mode='lines',
                          line=dict(width=0),
                          fill='tonexty',
                          fillcolor=f'rgba{tuple(int(float(x)*0.7) for x in colors[i][4:-1].split(",")) + (0.2,)}',
                          showlegend=False),
                row=(i // n_cols) + 1,
                col=(i % n_cols) + 1
            )

        # Plot mean waveform
        fig.add_trace(
            go.Scatter(x=time_vector, y=mean_waveform,
                      mode='lines',
                      line=dict(color=colors[i], width=2),
                      name=f'MU {i+1} (n={len(waveforms)})',
                      hovertemplate='Time: %{x:.1f} ms<br>Amplitude: %{y:.3f}<br>%{name}<extra></extra>'),
            row=(i // n_cols) + 1,
            col=(i % n_cols) + 1
        )

    # Update layout
    fig.update_layout(
        height=subplot_height * n_rows,
        width=subplot_width * n_cols,
        showlegend=True,
        plot_bgcolor='white',
        title='Motor Unit Action Potential Waveforms'
    )

    # Update axes
    fig.update_xaxes(
        title_text='Time (ms)',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray'
    )
    fig.update_yaxes(
        title_text='Amplitude',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray'
    )

    fig.show()


def plot_spike_train(spike_train, sampling_frequency, silhouette_scores=None,
                    min_score=0.93, **kwargs):
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

    # Set default parameters
    spike_height = kwargs.get('spike_height', 0.4)
    spike_width = kwargs.get('spike_width', 0.01)
    color_plot = kwargs.get('color_plot', True)
    colormap = kwargs.get('colormap', 'viridis')
    x_range = kwargs.get('x_range', None)
    target_height = kwargs.get('target_height', 800)
    units_per_height = kwargs.get('units_per_height', 40)

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
                               (selected_spikeTrain.shape[0] / sampling_frequency / 10))
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
