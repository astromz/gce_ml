#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper function that passes command line inputs to the actual trainer for either Cloud ML or GCE ML.
Supports both python2 and python3.

Must config `trainer_configs/trainer_config.yaml` first before running model.
All parameters and network architecture are configured in that file!

----------
Created on Mon Oct 19 2017
@author: Ming Zhao
"""
from __future__ import absolute_import, division, print_function
import argparse
import time, os
import yaml
from tensorflow.python.lib.io import file_io
from tensorflow import __version__ as tf_version

from trainer import mnist_autoencoder_deconv_simple


if __name__ =='__main__':
    # test write permission by writing something random to disk
    with open('testing_output.txt', 'w+') as f:
        f.write('Hello World!\n')
        f.write('This is a test written by custom startup script!\n')
        f.write('Success. Yay!\n')
        f.write('Current time = {}'.format(time.ctime()))
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('Current path: {}'.format(dir_path))

    # Now code that actually does stuff
    parser = argparse.ArgumentParser()

    # Required Arguments
    parser.add_argument(
      '--job_dir', help='GCS location to write checkpoints and export models', required=True)

    parser.add_argument(
      '--job_id', help='ID for the training job, passed from instance creation', required=True)

    parser.add_argument(
      '--config_file', help='A YAML config file that contrains all other trainer input parameters', required=True)

    parser.add_argument(
      '--data_path', help='GS path for training data', required=True)


    args = parser.parse_args()
    arguments = args.__dict__

    ########### Load config file and config parameters ################
    if args.config_file is None:
        raise ValueError('config YAML file must not be None!!!')
    if file_io.file_exists(args.config_file) is not True:
        # use tf's file_io for both GS and local files
        raise ValueError('config file does not exsit!!!  {}'.format(args.config_file))

    with file_io.FileIO(args.config_file, 'r') as f:  # This reads BOTH local files and GS bucket files!!!
        config = yaml.load(f)

    # actually training happens here
    mnist_autoencoder_deconv_simple.train(job_dir=args.job_dir, job_id=args.job_id,
                                          data_path=args.data_path, **config)
