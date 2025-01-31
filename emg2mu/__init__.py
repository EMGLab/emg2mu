"""
EMG2MU: A Python package for motor-unit decomposition on hdEMG datasets.

This package provides tools for analyzing and decomposing high-density EMG signals
into their constituent motor unit action potentials.

Main Features:
- Signal preprocessing with whitening and noise injection options
- ICA-based decomposition with CPU and GPU (PyTorch) implementations
- Duplicate motor unit detection and removal
- Quality assessment using silhouette scores
- Visualization tools for spike train analysis
- Comprehensive I/O utilities for data and results management

For more information, visit: https://github.com/neuromechanist/emg2mu
"""

from .core.decomposition import EMG
from .version import __version__

__all__ = ['EMG', '__version__']
