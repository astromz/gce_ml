#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper function that passes command line inputs to the actual trainer for GCloud ML.
Supports both python2 and python3.

Must config model_config_batch4_c.yaml first before running model.
All parameters and network architecture are configured in that file!

Example:
+ run locally inside cloud_ml/:
    `python cloud_ml/task.py --train_config_file model_config_batch4_c.yaml --train_data_path /Users/207229/Work/toner/data_batch4/ --job-dir /Users/207229//Work/toner/model_outputs/ `

+ run using cloud-ml-engine:
    `>cd ~/work/toner/source/cloud_ml`
    `> ./gcloud.local.run.sh`

----------
Created on Mon Jun 19 15:09:06 2017
@author: Ming Zhao
"""
from __future__ import absolute_import, division, print_function
import argparse

from trainer import mnist_autoencoder_deconv_simple


if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    # Argument required by GC
    parser.add_argument(
      '--job-dir', help='GCS location to write checkpoints and export models', required=True)

    parser.add_argument('--job_id', help='Job ID to tag models', required=True)

    parser.add_argument('--use_transposed_conv', help='Use "deconv" layers or transposed conv layers for "deconv"',
                        action='store_true', default=False)

    parser.add_argument('--score_metric', help='Metric for scoring: mse, mae, binary_entropy, etc.',
                        default='mse')

    parser.add_argument('--loss', help='Loss function: mse, mae, binary_crossentropy, etc.',
                        default='binary_crossentropy')

    parser.add_argument('-lr', '--learning_rate', help='learning rate',
                        default=0.001, type=float)

    parser.add_argument('--lr_decay', help='Learning rate decay (e.g., linear decay for Adma)',
                        default=0.001, type=float)

    parser.add_argument('-opt','--optimizer_name', help='optimizer function name',
                        default='adam')

    parser.add_argument('--n_epochs', help='Number of epochs',
                        default=100, type=int)

    parser.add_argument('--patience', help='Number of epochs to wait before early stopping',
                        default=5, type=int)

    parser.add_argument('--batch_norm_before_activation', help='Put batch_norm layer before activation',
                        default=False, action='store_true')

    parser.add_argument('--pool_method', help='Pooling method, either "max" or "average"',
                        default="max")

    args = parser.parse_args()
    arguments = args.__dict__

    mnist_autoencoder_deconv_simple.train(**arguments)
