# emg2mu: GPU-Accelerated High-Density EMG Decomposition

A PyTorch-accelerated toolkit for decomposing high-density EMG (hdEMG) signals into individual motor unit action potentials using blind source separation techniques.

## Features

- Fast motor unit decomposition using GPU-accelerated Independent Component Analysis (ICA)
- Efficient signal preprocessing including whitening and noise injection
- Automated duplicate motor unit removal
- Quality assessment using silhouette coefficients
- Interactive visualization of decomposed spike trains
- Compatible with common hdEMG datasets (e.g., Hyser dataset)

## Installation

```bash
# Download the repository from GitHub
git clone https://github.com/neuromechanist/emg2mu

# Install the package
cd emg2mu
pip install .
```

## Quick Start

```python
from emg2mu import EMG

# Load and decompose hdEMG data
emg = EMG('path/to/data.mat')
emg.preprocess()
emg.run_ICA(method='torch')  # Uses GPU acceleration
emg.remove_motorUnit_duplicates()
emg.compute_score()
emg.spikeTrain_plot()
```

## Example Dataset

The package includes a sample dataset in the `example` folder. You can run:

```python
from emg2mu import EMG

# Run decomposition on sample data
emg = EMG('example/sample1.mat')
emg.run_decomposition()
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

When using this toolbox, please cite:

```bibtex
@software{shirazi2024emg2mu,
  author = {Shirazi, Seyed Yahya},
  title = {emg2mu: GPU-accelerated High-Density EMG Decomposition},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/neuromechanist/emg2mu}
}
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This toolbox is based on the hdEMG decomposition work by Jiang et al. (2021) and implements the FastICA algorithm with GPU acceleration using PyTorch.

## Contact

Seyed (Yahya) Shirazi - [@neuromechanist](https://github.com/neuromechanist)

Project Link: [https://github.com/neuromechanist/emg2mu](https://github.com/neuromechanist/emg2mu)