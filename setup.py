import sys
import os
from setuptools import setup, find_packages

PACKAGES = find_packages()

# Get version and release info, which is all stored in emg2mu/version.py
ver_file = os.path.join('emg2mu', 'version.py')
with open(ver_file) as f:
    exec(f.read())

# Give setuptools a hint to complain if it's too old a version
# 24.2.0 added the python_requires option
# Should match pyproject.toml
SETUP_REQUIRES = ['setuptools >= 24.2.0']
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []

REQUIRES = [
    'numpy>=1.20.0',
    'scipy>=1.7.0',
    'scikit-learn>=1.0.0',
    'plotly>=5.0.0',
    # Optional for non-M1 Macs
    'torch>=2.5.0;platform_system!="Darwin" or platform_machine!="arm64"',
    # For M1 Macs
    'torch>=2.5.0;platform_system=="Darwin" and platform_machine=="arm64"'
]

PYTHON_REQUIRES = ">=3.8"

opts = dict(
    name='emg2mu',
    maintainer='Seyed Yahya Shirazi',
    maintainer_email='shirazi@ieee.org',
    description='Decompose hd-EMG signals into motor units',
    long_description='''This package uses a suite of blind source separation techniques to decompose 
    hd-EMG signals into motor units.''',
    url='https://github.com/neuromechanist/emg2mu',
    download_url='https://pypi.org/project/emg2mu',
    license='CC-BY-NC-SA-4.0',
    author='Seyed Yahya Shirazi',
    author_email='shirazi@ieee.org',
    platforms='any',
    version='0.0.1',
    packages=PACKAGES,
    install_requires=REQUIRES,
    python_requires=PYTHON_REQUIRES,
    setup_requires=SETUP_REQUIRES,
)

if __name__ == '__main__':
    setup(**opts)
