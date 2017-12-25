# Example to train Keras models on Googel Cloud ML

This example shows how to train a convolutional autoencoder using Keras and GPU on Cloud ML.
It also shows how to use Tensorboard in Keras and save files in Cloud Storage. 

Meanwhile the example allows the exploration of using "deconv" layers (i.e., transposed convolution), dilated convolution (atrous convolution), as well as comparing batch norm before and after a non-linear activation layer. 

## Prerequisite
Set up Google Cloud Platform. Follow this [link](https://cloud.google.com/ml-engine/docs/quickstarts/command-line) to:

  - Select or create a project on Google Cloud Platform
  - For first time users, enable billing. You can sign up for a free trial with $300 credits.
  - Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/quickstart-mac-os-x#before-you-begin) 
  - Initialize your gcloud environment at command line: ** `gcloud init` **
  	+ set up your email account, region (us-east is among the cheapest), etc.
  - And here is an [overview of the Cloud ML Engine](https://cloud.google.com/ml-engine/docs/concepts/technical-overview)
  - **Note**: Please follow the exact folder structure when making your own cloud ML package after trying this example. Change `setup.py` and `trainer.task.py` accordingly. For details, check [here](https://cloud.google.com/ml-engine/docs/images/recommended-project-structure.png) and [here](https://cloud.google.com/ml-engine/docs/how-tos/packaging-trainer)

## Training
 - To train model locally, run the following shell script on command line. It should take <10 min since `n_epochs` is set to only `3`.

		$ ./local.run.sh 

 - To train remotely on Google cloud ML engine, 
     + First make sure you have created a project (e.g., my_first_proj) and a bucket (a folder in cloud storage, e.g., my_bucket)
     + Change the environment variable `BUCKET_NAME` in `remote.run.sh` accordingly.
     + Make sure your `REGION` variable is set correctly
     + Change `--n_epochs` to say, 10, in `remote.run.sh`, for a quick test. You can change it to 100 later.
     + Run the following command:

  			$ ./remote.run.sh

  + Now follow the output instructions in the command line to either stream your log or check your log in GCP console.


That is it! You now should have successfully trained your first convolutional autoencoder on Cloud ML with 1 GPU. 

## Check your results
 - The code should automatically save two PDF figures, either on local disk or on gs bucket. You can check the learning curves there.
 - Alternatively, you can use `tensorboard` to examine your neural net graph, variables and more. Type the following and follow instructions.

		$ tensorboard --logdir=gs://my_project_name/my_bucket/job_id 

		
		 
