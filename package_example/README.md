### Example Python package to train Keras models with gce_ML
___
This example shows how to train a convolutional autoencoder using Keras and GPU on GCE ML.
It also shows how to use Tensorboard in Keras and save files in Cloud Storage.

Meanwhile the example allows the exploration of using "deconv" layers (i.e., transposed convolution), dilated convolution (atrous convolution), as well as comparing batch norm before and after a non-linear activation layer.

#### Prerequisite
Set up gce_ML by following this [link](https://github.com/astromz/gce_ml)/

#### Training
 - Configure your GCE submission script here. Then run:

  			$ ./submit_model.sh

  + Now follow the output instructions in the command line to either stream your log or check your instance in GCE console.


That is it! You now should have successfully trained your  convolutional autoencoder using the gce_ML scripts.

#### Check your results
 - The code should automatically save two PDF figures, either on local disk or on gs bucket. You can check the learning curves there.
 - Alternatively, you can use `tensorboard` to examine your neural net graph, variables and more. Type the following and follow instructions.

		$ tensorboard --logdir=gs://my_project_name/my_bucket/job_id
