#! /bin/bash

sleep 30
echo "Running start up script ..."

### Metadata specification
# All this metadata is pulled from the Compute Engine instance metadata server

# Your account username on the GCE instance
GCE_USER=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/gce_user -H "Metadata-Flavor: Google")

# Trainer Job name or ID in meta data
export JOB_ID=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/job_id -H "Metadata-Flavor: Google")

# DIR where all JOB data will reside
export JOB_DIR=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/job_dir -H "Metadata-Flavor: Google")

# trainer module name
export TRAINER_MODULE_NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainer_module_name -H "Metadata-Flavor: Google")

# TRAINER PACKAGE_PATH
export PACKAGE_PATH=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/package_path -H "Metadata-Flavor: Google")

# TRAINER_CONFIG_FILE
export TRAINER_CONFIG_FILE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/trainer_config_file -H "Metadata-Flavor: Google")

# TRAIN_DATA_PATH
export TRAIN_DATA_PATH=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/train_data_path -H "Metadata-Flavor: Google")

# KEEP_ALIVE=True then instance won't shut down after training is complete
KEEP_ALIVE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive -H "Metadata-Flavor: Google")


cd "/home/$GCE_USER"
pwd
echo "Current path: $(pwd)"
echo "Python path: $(which python)"
echo "Python Version:"
echo "$(python -V)"

# Install pip if not installed
if (which pip | grep pip &> /dev/null) ;then
  echo "Found pip"
else
  echo "pip not found! Installing pip"
  apt-get --assume-yes install python-pip python-dev build-essential
fi


# Download trainer package from gs://
echo "Downloading package: ${JOB_DIR}/${JOB_ID}/package/"
gsutil cp -r ${JOB_DIR}/${JOB_ID}/package/ ./

# Download config file
echo "Downloading config file: ${JOB_DIR}/${JOB_ID}/config/${TRAINER_CONFIG_FILE}"
gsutil cp  ${JOB_DIR}/${JOB_ID}/config/${TRAINER_CONFIG_FILE} ./
# send log to GS
gsutil cp -r /var/log/syslog ${JOB_DIR}/${JOB_ID}/logs/

sleep 3
pip install ./package/dist/*.tar.gz



######### Now run the python trainer job   ##############
echo "Now running custom trainer job : ${TRAINER_MODULE_NAME} ..."
# send log to GS
gsutil cp -r /var/log/syslog ${JOB_DIR}/${JOB_ID}/logs/

sudo -u $GCE_USER python -m $TRAINER_MODULE_NAME --job_dir ${JOB_DIR}/${JOB_ID}/ --job_id ${JOB_ID} --config_file ${TRAINER_CONFIG_FILE} --data_path ${TRAIN_DATA_PATH}

echo "Training job finished! "
echo


### Once the job has completed, keep alive for $KEEP_ALIVE seconds,
### then shut down the Compute Engine instance
echo "Sleeping for $KEEP_ALIVE seconds, then shut down."
# send log to GS
gsutil cp -r /var/log/syslog ${JOB_DIR}/${JOB_ID}/logs/

sleep $KEEP_ALIVE

# send log to GS
gsutil cp -r /var/log/syslog ${JOB_DIR}/${JOB_ID}/logs/

sudo shutdown -h now
