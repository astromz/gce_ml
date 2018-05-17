# Ubuntu Image
This is a [Packer](https://www.packer.io) template for building a GCE machine image with `CUDA9.0` and `cuDNN7` and `tensorflow-gpu 1.7` installed.

## Usage

1. Make sure packer is installed (`brew install packer`)
1. Download cuDNN deb installer [here](https://developer.nvidia.com/rdp/cudnn-download), select `Download cuDNN v7.0.5 (Dec 5, 2017), for CUDA 9.0`, and `cuDNN v7.0.5 Runtime Library for Ubuntu16.04 (Deb)`. Save the file as `libcudnn7_7.0.5.deb` in this directory.
1. Configure `project_id`, `zone` in `packer.json`
1. Run `packer build packer.json`
1. After process completes (takes around 10 minutes), image will be available as name `ubuntu-cuda9-tf17-{{timestamp}}` in GCP console ([link](https://console.cloud.google.com/compute/images)).

> This assumes `gcloud` is installed and authenticated. If not, first create a service account with `Compute Admin` role and generate a JSON private key ([link](https://console.cloud.google.com/iam-admin/serviceaccounts)). Add `account_file` in the `googlecompute` builder in `packer.json` and point to the JSON private key file.
