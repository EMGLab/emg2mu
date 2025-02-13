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

    # Plot waveforms with individual spikes shown
    emg.plot(
        plot_type='waveforms',
        min_score=0.93,
        plot_individual=True,  # Show individual spikes
        colormap='turbo'
    )

    # Plot waveforms with only mean and confidence intervals
    emg.plot(
        plot_type='waveforms',
        min_score=0.93,
        plot_individual=False,  # Hide individual spikes for cleaner view
        colormap='turbo'
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
    emg.plot(plot_type='waveforms', min_score=0.93)


if __name__ == '__main__':
    # main()
    # Uncomment to load saved results instead of recomputing
    load_saved_results()
