# gce_ML
This small package makes ML model training with Google Compute Engine (GCE) easy with simple model submission and automatic VM instance management. It is very similar to Google's [Cloud ML Engine](https://cloud.google.com/ml-engine/docs/technical-overview) but offers more flexibility for customization (e.g., GPU, CPU, and RAM configurations) and debugging.

### Why use this package?
There are three primary advantages:

  1. You can *pre-build your own image* with all the libraries you need, so *what you have in the cloud is what you have locally*. This could save you some headaches with different runtime versions provided by the Cloud ML Engine, and also shortens your instance startup time and the debugging cycle.

  2. *Debugging is more straightforward than Cloud ML.* You can log in to the instance to debug and diagnose any problems in the cloud environment (e.g., with specific GPUs and memory and so on), instead of having to debug locally, re-submit your job, wait for 6-10 min for a new instance to start, and iterate.

  3. *More affordable GPU power and VM resources.* Using GCE's [Preemptible VM](https://cloud.google.com/preemptible-vms/) instances can significantly reduce cost (50% off), as long as you save your model frequently and do not need your instances to run continuously. Even better, Google just announced [Preemptible GPUs](https://cloudplatform.googleblog.com/2018/01/introducing-preemptible-gpus-50-off.html), which will make GPUs more affordable as well. According to Google: "You can now attach NVIDIA K80 and NVIDIA P100 GPUs to Preemptible VMs for $0.22 and $0.73 per GPU hour, respectively. This is 50% cheaper than GPUs attached to on-demand instances, ..."

Setting up the package takes some time, but once you correctly configure it, you can submit and train your models in the same way as in Cloud ML.

### Prerequisite
+ Google Cloud Platform account and billing enabled. Follow this [link](https://cloud.google.com/ml-engine/docs/command-line)  
    - Select or create a project on Google Cloud Platform
    - For first time users, enable billing. You can sign up for a free trial with $300 credits.
    - Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/quickstart-mac-os-x#before-you-begin). We will use the `gcloud` CLI for our tasks.
    - Initialize your gcloud environment at command line: ** `gcloud init` **
    	+ set up your email account and region (us-east is among the cheapest).
    - For more details: here is an [overview of the Cloud ML Engine](https://cloud.google.com/ml-engine/docs/concepts/technical-overview)
    - **Note**: Please follow the exact folder structure when making your own cloud ML package after trying this example. Change `setup.py` and `trainer.task.py` accordingly. For details, check [here](https://cloud.google.com/ml-engine/docs/images/recommended-project-structure.png) and [here](https://cloud.google.com/ml-engine/docs/how-tos/packaging-trainer)


### Create a GCE instance and then a custom boot image
We will create a new GCE virtual machine instance using a public image and customize it. You can then create GCE instances using pre-existing custom images later. *This instruction is based on the example and steps from [here (Compute Engine survival training)](https://github.com/GoogleCloudPlatform/ml-on-gcp/tree/master/gce/survival-training) with modifications and a few more details.*

1. Create your instance. The easiest way is to use the Cloud Console [here](https://console.cloud.google.com/compute/). Follow this [link](https://cloud.google.com/compute/docs/instances/create-start-instance).
    - Feel free to play with the customization of CPUs, memory, disk storage, etc. For GPUs, see below.
    - For `Boot disk` image, choose `Ubuntu 16.04 LTS` for this exercise. Other images may cause compatibility issues. A disk size of >=20Gb should be enough.
    - To be able to use GPUs, you may need to enable your GPU quota [here](https://console.cloud.google.com/compute/quotas?).
    - For `Access scopes`, choose `Allow full access to all Cloud APIs` so your instance can read/write to Cloud Storage.
    - For `Firewall`, choose `Allow HTTPS traffic`.
    - **Before clicking the `Create` button**, you can click the blue `command line` link located below the `Create` button to see what the full `gcloud` command is for this instance you just configured. **I found this CLI feature a very convenient debugging tool**.
    - Now click `Create` to create your VM instance.
    - To check your instance:

          $ gcloud compute instances list

    - To log in to your instance:

          $ gcloud compute ssh my_instance --zone=us-east1-c


2. Add GPU(s) to your instance
    - As a beginner, try the cheapest GPU configuration by selecting `1x Tesla K80` GPU.
    - Follow this [link](https://cloud.google.com/compute/docs/gpus/add-gpus) to add GPUs.
    - Now, first initialzie your gcloud:
              $ gcloud init
    - Then, log in to your instance and install CUDA drivers:
        + Log in:
              $ gcloud config set project [my-project-id]
              $ gcloud compute ssh [my_instance] --zone=us-east-1c

        + Follow [this link](https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script) to install drivers. Click `UBUNTU` to see the script for `Ubuntu 16.04 LTS - CUDA 8` (Tensorflow does not support other driver versions yet!).
    - Note the [Optimizing GPU performance](https://cloud.google.com/compute/docs/gpus/add-gpus#gpu-performance) section.
    - Next, add the CUDA path to our `LD_LIBRARY_PATH` environment variable in `~/.bashrc` file on the VM, as per the TensorFlow instructions (also [outlined here](https://github.com/GoogleCloudPlatform/ml-on-gcp/blob/master/gce/survival-training/README-tf-estimator.md#cuda-drivers)). Add the following lines to the end of your `.bashrc`:

          export CUDA_HOME=/usr/local/cuda-8.0
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

      With that done, run

            $ source ~/.bashrc

      And to verify that the change took effect:

            $ echo $CUDA_HOME $LD_LIBRARY_PATH

    - Next, finish the rest of the steps below (also outlined in this [instruction](https://github.com/GoogleCloudPlatform/ml-on-gcp/blob/master/gce/survival-training/README-tf-estimator.md#cudnn-library):
      + Download cuDNN v6 for CUDA 8.0 from [here](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/cudnn-8.0-linux-x64-v6.0-tgz) as required for tensorflow (other version are not supported). You may need to first register an account with NVIDIA.
      + Upload your downloaded `.tgz` file to your VM instance:

            $ gcloud compute scp ~/Downloads/cudnn-8.0-linux-x64-v6.0.tgz [your-instance-name]:~/

      + Next, in your VM instance's terminal:

            $ tar xvfz cudnn-8.0-linux-x64-v6.0.tgz
            $ sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
            $ sudo cp cuda/lib64/* $CUDA_HOME/lib64
            $ sudo chmod a+r /usr/local/cuda-8.0/include/cudnn.h $CUDA_HOME/lib64/libcudnn*
            $ sudo apt-get install libcupti-dev


3. Create a pre-configured boot image

    - Custom images allow you to create new instances with the same state you configured, so you do not have to re-install CUDA drivers and python packages, etc. That is, you only need to do these configurations and setups once.

    - First, install all required **Python (2.7)** packages and libraries in your instance (e.g., pip, numpy, pandas, matplotlib, scipy, sklearn, etc.)

      + To do this, we must install everything under `root`  to make the automatic model submission and training feature work, as the startup script is executed by `sudo`. Now, log in as `root` in your instance by typing:

            $ sudo -s

        Now you can install `pip` and other developer tools:

            $ apt-get --assume-yes install python-pip python-dev build-essential

        Now, install `tensorflow-gpu` for your GPU-enabled instance:

            $ pip install tensorflow-gpu==14.0

        Then, you can install other Python libraries one by one; or you can install them all together using a `requirements.txt` file.

            $ pip install -r requirements.txt

      + Once done, enter a `Python` interpreter and verify that your installed packages work:

            > # This is your python environment
            > import tensorflow as tf
            > tf.Session()
            > # you should see messages like:
            > # Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)


      + **NOTE**: here we use the default **Python 2.7** available as `root`. You can install your own Python version as long as you do it under root and it does not require running a `.bashrc` in shell (e.g., **Anaconda's Python distribution requires sourcing a bash script and thus won't work under sudo**).
      You can check your set up and python version using:

            $ sudo python -V

    - Now, your instance is fully configured. Exit your instance (`exit`) and get back to your terminal. Stop it to create a boot image.

          $ gcloud compute instances stop my-instance
          $ gcloud compute images create my-boot-image --source-disk my-instance-name --source-disk-zone us-east1-c

      That is it! You now have a boot image to create other instances with exactly the same state (GPU configurations, python libraries, etc.). Also, you can [share your image among other projects](https://cloud.google.com/compute/docs/images/sharing-images-across-projects).

4. Set up Cloud Storage (GCS) for all your data and files
    - Like Cloud ML, we store all data, logs, and model checkpoints on Cloud Storage buckets. You will need to create a bucket for this project beforehand (if you have not). Just follow this [link](https://cloud.google.com/storage/docs/creating-buckets) and create a bucket (e.g., `my_bucket`).
    - Upload your data to your GCS bucket for later access. You can upload data to a new bucket or the bucket just created.
    - Now, the `JOB_DIR` variable that you will need later is: `gs://project_name/my_bucket/`. Each submitted model automatically creates a subfolder inside it.


### Install and set up gce_ml, then train your models at scale
Finally, we are ready to set up this package and train your models.

1. Clone this package to your local directory.

2. Set up your model in python
    - Use the autoencoder-decoder model in `package_example/` as an example.
    - Your training package should be constructed in pretty much the same way as in Cloud ML instances. Follow the exact folder structure listed [here](https://cloud.google.com/ml-engine/docs/images/recommended-project-structure.png) and [here](https://cloud.google.com/ml-engine/docs/how-tos/packaging-trainer) when making your own cloud ML package. Change `setup.py` and `trainer.task.py` accordingly.
    - **NOTE 1:** There is, however, one small difference between this package and Cloud ML -- your model input variables are supplied by an external `YAML` configuration file instead of using `bash` commands. This approach actually makes your training easier to manage, as once you set up your gce_ml submission script, the only things you need to change are the new instance name (actually not necessary if using timestamp as instance name) and the `YAML` configuration file (for different model parameters).
    - **NOTE 2:** You can rename your package to your liking, but make sure the folder `gce_scripts/` exists and resides at the same level as your `submis_model.sh` script (again, follow the directory structure of the package).  

2. Model training with **gce_ML**
    - Now you have gone through all setup steps and are finally ready to submit a model for training. Follow the example in  `submit_model_gpu.sh` or `submit_model.sh` to configure your new instance.
    - To submit your model to a new GCE instance, just run:

          $ ./submit_model.sh

    - You can monitor the instance by going to the [GCE console](https://console.cloud.google.com/compute/instances?project=) and click the newly created instance. Alternatively, you can follow the instruction shown in the shell prompt to stream your `syslog` to your terminal.

          $ gcloud compute instances tail-serial-port-output my-instance-with-gpu --port 1

    - Your instance **will automatically shut down** once completed and can be restarted later. If needed, you can keep it alive for a given number of seconds so you can log in and debug.
    - Finally, **remember to delete your instance** from either the [Cloud Console](https://console.cloud.google.com/compute/instances?project=) or command line. It automatically shuts down but does not delete itself from the cloud. Each instance (and its associated job) is supposed to have a unique name, so you will not need it afterward (reuse will not save resource).

          $ gcloud compute instances delete my-instance

    - Don't worry, your models, checkpoints, and `syslog` are all saved in your job's unique GCS bucket (again, like Cloud ML). You can find all relevant information on GCS as long as you save your model outputs there by following the example.
