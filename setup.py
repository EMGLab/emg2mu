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

# opts = dict(name='emg2mu',
#             maintainer='Seyed Yahya Shirazi',
#             maintainer_email='shirazi@ieee.org',
#             description='Decompose hd-EMG signals into motor units',
#             long_description='This packege uses a suite of blind source separation techniques to decompose hd-EMG signals into motor units.'
#             url='github.com/neuromechanist/emg2mu',
#             download_url='pypi.org/project/emg2mu',
#             license='MIT',
#             # classifiers=CLASSIFIERS,
#             author='Seyed Yahya Shirazi',
#             author_email='shirazi@ieee.org',
#             platforms='any',
#             version='0.0.1',
#             packages=PACKAGES,
#             package_data=PACKAGE_DATA,
#             install_requires=REQUIRES,
#             python_requires=PYTHON_REQUIRES,
#             setup_requires=SETUP_REQUIRES,
#             requires=REQUIRES)


if __name__ == '__main__':
    setup(**opts)
