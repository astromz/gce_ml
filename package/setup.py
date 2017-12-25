#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ming Zhao
"""

# setup.py
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
                     'pyYaml>=3.12',
                     'keras>=2.0',
                     'matplotlib',
		             'h5py'#,
                     #'tensorflow>=1.2'  # do not include if has custom compiled version   
                     ]


setup(
    name='mnist_ae_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    #package_data={'sample': ['package_data.txt']}, # OPTIONAL
    include_package_data=True,
    description='Example trainer package for MNIST convolutional autoencoder',
    author='mz',
    author_email='ming.zhao@nytimes',
    zip_safe=True,
    url=' ' # Required. You will get a warning of missing meta-data if not provided.
)
