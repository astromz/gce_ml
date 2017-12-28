#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    ####### Instance parameters #######
    --gce_username)
    GCE_USER="$2"
    shift # past argument
    shift # past value
    ;;
    --project)
    PROJ_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    --instance)
    INSTANCE_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    --image)
    BOOT_IMAGE="$2"
    shift # past argument
    shift # past value
    ;;
    --image-project)
    image_project="$2"
    shift # past argument
    shift # past value
    ;;
    --machine-type)
    machine_type="$2"
    shift # past argument
    shift # past value
    ;;
    --maintenance-policy)
    maintenance_policy="$2"
    shift # past argument
    shift # past value
    ;;
    --accelerator)
    accelerator="$2"
    shift # past argument
    shift # past value
    ;;
    --min-cpu-platform)
    min_cpu_platform="$2"
    shift # past argument
    shift # past value
    ;;
    --tags)
    tags="$2"
    shift # past argument
    shift # past value
    ;;
    --boot-disk-size)
    boot_disk_size="$2"
    shift # past argument
    shift # past value
    ;;
    --job_id)
    job_id="$2"
    shift # past argument
    shift # past value
    ;;
    --job_dir)
    job_dir="$2"
    shift # past argument
    shift # past value
    ;;
    --trainer_module)
    trainer_module_name="$2"
    shift # past argument
    shift # past value
    ;;
    --trainer_package_path)
    trainer_package_path="$2"
    shift # past argument
    shift # past value
    ;;
    --zone)  # Optional
    ZONE="$2"
    shift # past argument
    shift # past value
    ;;
    --keep_alive)  # Optional
    KEEP_ALIVE="$2"
    shift # past argument
    ;;
    ####### model parameters #######
    --trainer_config)
    trainer_config_file="$2"
    shift # past argument
    shift # past value
    ;;
    --train_data_path)
    train_data_path="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

ZONE="${ZONE:-us-east1-c}"

KEEP_ALIVE="${KEEP_ALIVE:-0}"


echo
echo "Input GCE Parameters:"
echo "-------------------------------------------------"
echo "New Instance Name    =  ${INSTANCE_NAME}"
echo "Unique Job ID        =  ${job_id}"
echo "Job Dir on GS        =  ${job_dir}"
echo "GCE USER NAME        =  ${GCE_USER} (username for your account)"
echo "PROJ_NAME            =  ${PROJ_NAME}"
echo "Boot Image Name      =  ${BOOT_IMAGE}"
echo "Boot Image Project   =  ${image_project}"
echo "Machine Type         =  ${machine_type}"
echo "GPU (accelerator)    =  ${accelerator}"
echo "Boot Disk Size       =  ${boot_disk_size}"
echo "min_cpu_platform     =  ${min_cpu_platform}"
echo "ZONE                 =  ${ZONE} (us-east1-c is recommended)"
echo "KEEP_ALIVE when done =  ${KEEP_ALIVE} (seconds to keep alive when done, then shutdown)"
echo
echo
echo "Input Trainer Parameters:"
echo "-------------------------------------------------"
echo "trainer_module_name  =  ${trainer_module_name} (trainer module to execute)"
echo "trainer_package_path =  ${trainer_package_path} (local path)"
echo "train_config_file    =  ${trainer_config_file}"
echo "data_path on GS      =  ${train_data_path}"
echo

while true; do
  read -p "Please check your inputs. Ready to proceed? (yes/no, y/n)" yn
  case $yn in
      [Yy]* ) break;;
      [Nn]* ) exit;;
      * ) echo "Please answer yes or no.";;
  esac
done
echo



############# CHECKING AND SETTING PARAMETERS #####################

# Check startup.sh script
STARTUP="./gce_scripts/gce_ml_custom_startup.sh"
if !ls ${STARTUP} &> /dev/null
then
  echo "STARTUP.sh NOT FOUND! Make sure you have '$STARTUP'.  EXIT"
  exit 3 # local file not found error
fi

# set default compute zone. Best choice = us-east1-c (as of 12/2017)
gcloud config set compute/zone $ZONE &> /dev/null

# Original project the current console was in
original_proj=$(gcloud config get-value project)
echo "Current Project = $original_proj"
if [ "$original_proj" != "$PROJ_NAME" ]
then
  # set project id for instance
  gcloud config set project $PROJ_NAME &> /dev/null
  echo "Switching to project = $PROJ_NAME"
fi

# Check network and subnet
if gcloud compute networks list --filter="$PROJ_NAME-net" | grep "$PROJ_NAME-net" &> /dev/null
then
  network=$PROJ_NAME-net
  echo "Found network: $network"
else
  echo "Network NOT found: $network! EXIT" ;
  exit 5 # network not found
fi

if gcloud compute networks subnets list --filter="network:$PROJ_NAME-net" | grep "$PROJ_NAME" &> /dev/null
then
  subnet=$(gcloud compute networks subnets list --filter="network:$PROJ_NAME-net" | grep -m 1 "$PROJ_NAME" | cut -d' ' -f 1 | head -1)
  echo "Found subnet: $subnet"
else
  echo "Subnet NOT found! EXIT" ;
  exit 5 # network not found
fi


# Check image Source
if gcloud compute --project "$image_project" images list --filter=$BOOT_IMAGE | grep $BOOT_IMAGE &> /dev/null
then echo "Found boot image: $BOOT_IMAGE"
else
  echo "BOOT IMAGE $BOOT_IMAGE DOES NOT EXIST IN PROJECT $image_project !!!"
  exit 1 # boot image not found error
fi

# check data path on GS
if gsutil ls $train_data_path &> /dev/null
then echo "Found data path: $train_data_path"
else
  echo "GS DATA OR PATH NOT FOUND: $train_data_path"
  exit 2  # GS file not found error
fi

# check trainer package
if ls ${trainer_package_path} &> /dev/null
then echo "Found trainer package in: $trainer_package_path "
else
  echo "TRAINER PACKAGE NOT FOUND: $trainer_package_path"
  exit 3 # local file not found error
fi

# check trainer config file
if (ls $trainer_config_file) &> /dev/null
then echo "Found trainer config file : $trainer_config_file"
else
  echo "TRAINER CONFIG FILE NOT FOUND : $trainer_config_file"
  exit 3 # local file not found
fi

# If given instance is existant and running, then stop; elif existant then restart; else create instances
if (gcloud compute instances list --filter="name=$INSTANCE_NAME AND -status=TERMINATED" | grep $INSTANCE_NAME &> /dev/null) &> /dev/null
then
  echo "INSTANCE $INSTANCE_NAME ALREADY EXISTS AND IS ACTIVE. EXIT."
  exit 4 # Instance exists and is already active
  #restart=true
  #create=false
elif (gcloud compute instances list --filter="name=$INSTANCE_NAME AND status=TERMINATED" | grep $INSTANCE_NAME &> /dev/null) &> /dev/null
then
  echo "INSTANCE ALREADY EXISTS BUT TERMINATED : $INSTANCE_NAME "
  echo "Solutions: change the instance name, or delete your existing instance using 'gcloud compute instances delete $INSTANCE_NAME' "
  echo
  exit

  restart=true
  create=false
else
  echo "INSTANCE DOES NOT EXIST YET. WILL CREATE : $INSTANCE_NAME"
  create=true
  restart=false
fi



################# Build and upload package #####################
echo
echo 'Building and uploading package ...'
pushd $trainer_package_path &> /dev/null
python setup.py -q sdist --formats=gztar

gsutil cp -r dist/  ${job_dir}/${job_id}/package/

rm -rf dist *egg-info

popd &> /dev/null  # go back to parent dir
echo

# upload config yaml file, and trim path from filename for GCE
gsutil cp ${trainer_config_file}  ${job_dir}/${job_id}/config/
trainer_config_file_base=$(basename $trainer_config_file)
echo



################# Create or restart instance ##################

#if [ $create == false ] && [ $restart == true ]
#then
  # Adding or update metadata
#  echo '--> Updating metadata ... '
#  gcloud compute instances add-metadata $INSTANCE_NAME \
#--metadata job_id=$job_id,\
#job_dir=$job_dir,\
#trainer_module_name=$trainer_module_name,\
#trainer_package_path=$trainer_package_path,\
#train_config_file=$trainer_config_file,\
#train_data_path=$train_data_path,\
#keep_alive=$KEEP_ALIVE \
#--metadata-from-file startup-script=startup.sh # can't have space at the beginning for broken lines

#  echo '--> Restarting instance ...'
#  gcloud compute instances start $INSTANCE_NAME

if [ $create == true ] && [ $restart == false ]
then
  if [ -z ${accelerator+x} ]   # if var accelerator is set
  then
    echo '--> Creating new instance WITHOUT GPU...'

    gcloud compute --project "$PROJ_NAME" instances create "$INSTANCE_NAME" --image "$BOOT_IMAGE" \
--network "$network" --subnet "$subnet" --zone "$ZONE" \
--scopes "https://www.googleapis.com/auth/cloud-platform" \
--maintenance-policy "$maintenance_policy" --tags "https-server" --image-project=$image_project \
--machine-type "$machine_type" --min-cpu-platform "$min_cpu_platform" \
--boot-disk-size="$boot_disk_size" --boot-disk-type="pd-standard" \
--boot-disk-device-name="$INSTANCE_NAME" \
--metadata job_id=$job_id,\
job_dir=$job_dir,\
trainer_module_name=$trainer_module_name,\
package_path=$trainer_package_path,\
trainer_config_file=$trainer_config_file_base,\
train_data_path=$train_data_path,\
gce_user=$GCE_USER,\
keep_alive=$KEEP_ALIVE \
--metadata-from-file startup-script=$STARTUP # can't have space at the beginning for broken lines

  else
    echo '--> Creating new instance with GPU ...'

    gcloud compute --project "$PROJ_NAME" instances create "$INSTANCE_NAME" --image "$BOOT_IMAGE" \
--network "$network" --subnet "$subnet" --zone "$ZONE" \
--scopes "https://www.googleapis.com/auth/cloud-platform" \
--maintenance-policy "$maintenance_policy" --tags "https-server" --image-project=$image_project \
--machine-type "$machine_type" --min-cpu-platform "$min_cpu_platform" \
--boot-disk-size="$boot_disk_size" --boot-disk-type="pd-standard" \
--boot-disk-device-name="$INSTANCE_NAME" \
--accelerator $accelerator \
--metadata job_id=$job_id,\
job_dir=$job_dir,\
trainer_module_name=$trainer_module_name,\
package_path=$trainer_package_path,\
trainer_config_file=$trainer_config_file_base,\
train_data_path=$train_data_path,\
gce_user=$GCE_USER,\
keep_alive=$KEEP_ALIVE \
--metadata-from-file startup-script=$STARTUP # can't have space at the beginning for broken lines

  fi
fi

echo
echo "For degbugging, try 'gcloud compute ssh $INSTANCE_NAME' to log in to the created instance. \
You must debug as root by typing 'sudo -s'. Startup logs can be found with 'cat /var/log/syslog | grep startup-script' "
echo "To stream startup logs, try 'gcloud compute instances tail-serial-port-output $INSTANCE_NAME --port 1' "
echo

# set project for instance
#gcloud config set project $original_proj
#echo "Switching BACK to original project = $PROJoriginal_proj_NAME"
