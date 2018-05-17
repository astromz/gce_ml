#!/bin/bash

# Update packages
sudo apt update
sudo apt full-upgrade -y

# Install CUDA apt repo
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt install -y gnupg-curl # HTTPS support for gnupg
sudo sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt update

# Install CUDA
sudo apt install -y cuda-9-0

# cuDNN
sudo dpkg -i /tmp/libcudnn7_7.0.5.deb

# Tensorflow
sudo apt install -y python3-pip
sudo pip3 install tensorflow-gpu==1.7.0
