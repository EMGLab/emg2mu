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

## Sample Dataset

The package includes a sample dataset in the `sample_data` folder. You can run:

```python
from emg2mu import EMG

# Run decomposition on sample data
emg = EMG('emg2mu/sample_data/sample1.mat')
emg.run_decomposition()
```

This would be the spike-train plot from the sample data:
![newplot](https://github.com/user-attachments/assets/daca81f2-6832-4c27-bdd0-df6587acdac0)

Unfortunately, the sample ICA decomposition is not included due to the file size limitation. However, using the PyTorch-accelerated ICA algorithm, you can decompose the sample data in a couple of minutes.

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

This toolbox is based on the [hdEMG decomposition toolbox](https://github.com/neuromechanist/hdEMG-Decomposition) in Matlab and introduces a *Pythonic* implementation of the algorithm with the GPU-accelerated FastICA algorithm and vectorized silhouette scoring algorithm.

Note: The motor-unit scoring mechanism uses a different algorithm than the one used in the Matlab counterpart, resulting in overall higher scores for similar decompositions. However, I believe that this one is more transparent, scalable, and extensible. Please keep in mind the change.

## Contact

Seyed (Yahya) Shirazi - [@neuromechanist](https://github.com/neuromechanist)

Project Link: [https://github.com/neuromechanist/emg2mu](https://github.com/neuromechanist/emg2mu)
