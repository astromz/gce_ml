# gce_ML
This small package makes ML model training with Google Compute Engine (GCE) easy with simple model submission and automatic VM instance management. This is very similar to the [Cloud ML Engine](https://cloud.google.com/ml-engine/docs/technical-overview) but offers more flexibility for customization (e.g., GPU, CPU, and RAM configurations) and debugging.

### Why use this package?
There are two primary advantages:

  1. You can pre-build your own image with all the libraries you need, so what you'll have in the cloud is what you have locally. This could save you some headaches with different runtime versions provided by the Cloud ML Engine, and also shortens your instance startup time and the debugging cycle.

  2. Debugging is easier. You can log in to the instance to debug and diagnose any problems in the cloud environment (e.g., with specific GPUs and memory and so on), instead of having to debug locally, re-submit your job, wait for 6-10 min for a new instance to start, and iterate.

Setting up the package takes some time, but once it is properly configured, you can submit and train your models in the same way as in Cloud ML.

### Prerequisite
+ Google Cloud Platform account and billing enabled. Follow this [link](https://cloud.google.com/ml-engine/docs/command-line)  
    - Select or create a project on Google Cloud Platform
    - For first time users, enable billing. You can sign up for a free trial with $300 credits.
    - Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/quickstart-mac-os-x#before-you-begin). We will use the `gcloud` CLI for our tasks.
    - Initialize your gcloud environment at command line: ** `gcloud init` **
    	+ set up your email account, region (us-east is among the cheapest), etc.
    - For more details: here is an [overview of the Cloud ML Engine](https://cloud.google.com/ml-engine/docs/concepts/technical-overview)
    - **Note**: Please follow the exact folder structure when making your own cloud ML package after trying this example. Change `setup.py` and `trainer.task.py` accordingly. For details, check [here](https://cloud.google.com/ml-engine/docs/images/recommended-project-structure.png) and [here](https://cloud.google.com/ml-engine/docs/how-tos/packaging-trainer)


### Create a GCE instance and then a custom boot image
We will create a new GCE virtual machine instance using a public image and customize it. You can then create GCE instances using pre-existing custom images later. *This instruction is based on the example and steps from [here (Compute Engine survival training)](https://github.com/GoogleCloudPlatform/ml-on-gcp/tree/master/gce/survival-training) with modifications and a few more details.*

1. Create your instance. The easiest way is to use the Cloud Console [here](create the Compute Engine instance through the Cloud Console). Follow this [link](https://cloud.google.com/compute/docs/instances/create-start-instance).
    - Feel free to play with the customization of CPUs, memory, disk storage, etc. For GPUs, see below.
    - For `Boot disk` image, choose `Ubuntu 16.04 LTS` for this exercise. Other images may cause compatibility issues.
    - For `Access scopes`, choose `Allow full access to all Cloud APIs` so your instance can read/write to Cloud Storage.
    - For `Firewall`, choose `Allow HTTPS traffic`
    - **Before clicking the `Create` button**, you can click the blue `command line` link located below the `Create` button to see what the full `gcloud` command is for this instance you just configured. **I found this CLI feature is a very helpful debugging tool**.
    - Now click `Create` to create your VM instance.
    - To check your instance:

          $ gcloud compute instances list

    - To log in to your instance:

          $ gcloud compute ssh my_instance --zone=us-east1-c


2. Add GPU(s) to your instance
    - As a beginner, try the cheapest GPU configuration by selecting `1x Tesla K80` GPU.
    - Follow this [link](https://cloud.google.com/compute/docs/gpus/add-gpus) to add GPUs.
    - Now, log in to your instance and install CUDA drivers:
        + Log in:

              $ gcloud compute ssh [my_instance] --zone=us-east-1c

        + Follow [this link](https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script) to install drivers. Click `UBUNTU` to see the script for `Ubuntu 16.04 LTS - CUDA 8` (other driver versions are not supported!).
    - Note the [Optimizing GPU performance](https://cloud.google.com/compute/docs/gpus/add-gpus#gpu-performance) section.
    - Next, as per the TensorFlow instructions (also [outlined here](https://github.com/GoogleCloudPlatform/ml-on-gcp/blob/master/gce/survival-training/README-tf-estimator.md#cuda-drivers)), add the CUDA path to our `LD_LIBRARY_PATH` environment variable in `~/.bashrc` file on the VM. Add the following lines to the end of your .bashrc:

          export CUDA_HOME=/usr/local/cuda-8.0
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

      With that done, run

            $ source .bashrc

      And to verify that the change took effect:

            $ echo $CUDA_HOME $LD_LIBRARY_PATH

    - Next, finish the rest of the steps by following the instructions [here](https://github.com/GoogleCloudPlatform/ml-on-gcp/blob/master/gce/survival-training/README-tf-estimator.md#cudnn-library).


3. Create a pre-configured boot image

    - Custom images allow you to create new instances with exactly the state you configured, so you won't have to re-install CUDA drivers and python packages, etc. That is, you only need to do these configurations and setups once.

    - First, install all required **Python (2.7)** packages and libraries in your instance (e.g., pip, numpy, pandas, matplotlib, scipy, sklearn, etc.)

      + To do this, we must install everything under `root` in order to make the automatic model submission and training feature work, as the startup script was executed by `sudo`. Now, log in as `root` in your instance by typing:

            $ sudo -s

        Now you can install `pip` and other developer tools:

            $ apt-get --assume-yes install python-pip python-dev build-essential

        Now, install `tensorflow-gpu` for your GPU-enabled instance:

            $ pip install tensorflow-gpu

        Then, you can install other Python libraries one by one; or you can install them all together using a `requirement.txt` file.

      + Once done, enter a Python interpreter and verify that your installed packages work.  
      + **NOTE**: here we use the default **Python 2.7** available as `root`. You can install your own Python version as long as you do it under root and it does not require running a `.bashrc` in shell (e.g., **Anaconda's Python distribution requires sourcing a bash script and thus won't work under sudo**).
      You can check your set up and python version using:

            $ sudo python -V

    - Now, your instance is fully configured. Exit your instance (`exit`) and get back to your terminal. Stop it to create a boot image.

          $ gcloud compute instances stop my-instance
          $ gcloud compute images create my-boot-image --source-disk my-instance --source-disk-zone us-east1-c

      That is it. You now have a boot image to create other instances with exactly the same state (GPU configurations, python libraries, etc.). And you can [share your image among other projects](https://cloud.google.com/compute/docs/images/sharing-images-across-projects).

4. Set up Cloud Storage (GCS) for all your data and files
    - Like Cloud ML, we will store all data, logs, and model checkpoints on Cloud Storage buckets. You will need to create a bucket for this project beforehand (if you haven't). Just follow this [link](https://cloud.google.com/storage/docs/creating-buckets) and create a bucket (e.g., `my_bucket`).
    - Upload your data to your GCS bucket for later access. You can upload data to a new bucket or the bucket just created.
    - Now, the `JOB_DIR` variable that you will need later is: `gs://project_name/my_bucket/`. Each submitted model will automatically create a subfolder inside it.


### Install and set up gce_ml, then train your models at scale
Finally, we are ready to set up this package and train your models.

1. Clone this package to your local directory.

2. Set up your model in python
    - Use the autoencoder-decoder model in `package_example/` as an example.
    - Your training package should be constructed in pretty much the same way as in Cloud ML instances. Follow the exact folder structure listed [here](https://cloud.google.com/ml-engine/docs/images/recommended-project-structure.png) and [here](https://cloud.google.com/ml-engine/docs/how-tos/packaging-trainer) when making your own cloud ML package. Change `setup.py` and `trainer.task.py` accordingly.
    - **NOTE 1:** There is however one small difference between this package and Cloud ML -- your model input variables are supplied by an external `yaml` configuration file instead of using `bash` commands. This actually makes your training easier to manage, as once you set up your gce_ml submission script, the only things you need to change are the new instance name (actually not necessary if using timestamp as instance name) and the `yaml` configuration file (for different model parameters).
    - **NOTE 2:** You can rename your package to your liking, but make sure the folder `gce_scripts/` exists and resides at the same level as your `submis_model.sh` script (again, follow the directory structure of the package).  

2. Model training with **gce_ML**
    - Now you have gone through all setup steps and are ready to actually submit a model for training. Follow the example in  `submit_model_gpu.sh` or `submit_model.sh` to configure your new instance.
    - To actually submit your model to a new GCE instance, just run:

          $ ./submit_model.sh

    - You can monitor the instance by going to the [GCE console](https://console.cloud.google.com/compute/instances?project=) and click the newly created instance. Alternatively, you can follow the instruction shown in shell prompt to stream your `syslog` to your terminal.

          $ gcloud compute instances tail-serial-port-output my-instance-with-gpu --port 1

    - Your instance **will automatically shut down** once completed, and can be restarted later. If needed, you can keep it alive for a given number of seconds so you can log in and debug.
    - Finally, **remember to delete your instance** from either the [Cloud Console](https://console.cloud.google.com/compute/instances?project=) or command line. It will automatically shut down but won't delete itself from the cloud. Each instance (and its associated job) is supposed to have a unique name so you don't really need it afterwards (reuse won't save resource).

          $ gcloud compute instances delete my-instance

    - Don't worry, your models, checkpoints, and `syslog` are all saved in your job's unique GCS bucket (again, like Cloud ML). You can find all relevant information on GCS as long as you save your model outputs there by following the example.
