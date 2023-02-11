from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 1  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Neurscience"]

# Description should be a one-liner:
description = "emg2mu: Decmpose hd-EMG signals into motor units"
# Long description will go up on the pypi page
long_description = """
This packege uses a suite of blind source sepration techniques to decompose/
    hd-EMG signals into motor units.
The packekge takes advantage of pyTorch scaling to CPU or GPU platforms.
The package is still under development.
"""

NAME = "emg2mu"
MAINTAINER = "Seyed Yahya Shirazi"
MAINTAINER_EMAIL = "shirazi@ieee.org"
DESCRIPTION = 'Decompose hd-EMG signals into motor units'
LONG_DESCRIPTION = 'This package uses a suite of blind source separation techniques to decompose hd-EMG signals into motor units.'
URL = "http://github.com/neuromechanist/emg2mu"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Graham Findlay"
AUTHOR_EMAIL = "gfindlay@wisc.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'emg2mu': [pjoin('data', '*')]}
REQUIRES = ["numpy"]
PYTHON_REQUIRES = ">= 3.10"
