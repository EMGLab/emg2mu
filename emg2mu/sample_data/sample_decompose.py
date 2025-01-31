#!/usr/bin/env python3
"""
Sample script demonstrating EMG decomposition with waveform visualization.
"""

import os
import emg2mu

# Define paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, 'sample1.mat')
output_file = os.path.join(script_dir, 'sample1_decomposed')
ica_file = os.path.join(script_dir, 'sample1_ica.npz')
scores_file = os.path.join(script_dir, 'sample1_scores.npz')


def main():
    # Initialize EMG object with data
    emg = emg2mu.EMG(
        data=data_file,
        output_file=output_file
    )

    # Preprocess and run ICA
    emg.preprocess()
    emg.run_ica(
        method='torch',
        save_path=ica_file  # Save ICA results for future use
    )

    # Remove duplicates and compute scores
    emg.remove_duplicates()
    emg.compute_scores(save_path=scores_file)  # Save scores for future use

    # Plot spike trains
    emg.plot(
        plot_type='spike_train',
        spike_width=0.01,
        spike_height=0.4,
        color_plot=True,
        colormap='turbo',
        min_score=0.93
    )

    # Plot waveforms in grid layout (5 columns by default)
    emg.plot(
        plot_type='waveforms',
        min_score=0.93,
        window_size=0.005,  # ±5ms window
        plot_individual=True,  # Show individual spikes
        confidence_interval=True,
        alpha=0.1,
        colormap='turbo',
        n_cols=5,  # Number of columns in grid
        subplot_height=200,  # Height of each subplot in pixels
        subplot_width=300  # Width of each subplot in pixels
    )

    # Example of custom grid layout (e.g., 3 columns)
    emg.plot(
        plot_type='waveforms',
        min_score=0.93,
        window_size=0.005,
        plot_individual=False,  # Hide individual spikes for cleaner view
        confidence_interval=True,
        colormap='turbo',
        n_cols=3,  # Wider subplots with fewer columns
        subplot_height=250,
        subplot_width=400
    )


def load_saved_results():
    """
    Example of loading previously saved results.
    """
    emg = emg2mu.EMG(data=data_file)
    emg.run_ica(load_path=ica_file)  # Load saved ICA results
    emg.remove_duplicates()
    emg.compute_scores(load_path=scores_file)  # Load saved scores

    # Plot results
    emg.plot(plot_type='spike_train', min_score=0.93)
    emg.plot(
        plot_type='waveforms',
        min_score=0.93,
        n_cols=5,  # Use default grid layout
        subplot_height=200,
        subplot_width=300
    )


if __name__ == '__main__':
    # main()
    # Uncomment to load saved results instead of recomputing
    load_saved_results()
