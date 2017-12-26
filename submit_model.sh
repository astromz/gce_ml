#!/bin/bash

new_instance_name="test-cpu1"  # unique name for new instance
gce_username='my_gce_username' # replace with your account username for gce instance 
project="my-proj-dev"          # project-id
boot_image_name="test-image-2cpu-6gb" # pre-created boot image
image_project=$project

# VM Instance configurations. Use GCE console's `Create Instance` for reference
machine_type="custom-2-5120" # 2 CPUs, 5.120Gb RAM
maintenance_policy="MIGRATE" # MIGRATE for cpu; TERMINATE for gpu
min_cpu_platform='Intel Broadwell'
boot_disk_size="50"          # 50 Gb
zone="us-east1-c"


# Trainer Variables
job_id=$new_instance_name          # unique job id 
job_dir="gs://my-proj/test_dir"    # parent dir on GCS
trainer_module="trainer.task"
trainer_package_path="./package/"
trainer_config="./trainer_configs/trainer_config.yaml"
train_data_path="gs://my-proj/"    # data need to be pre-uploaded to GCS
keep_alive=700                     # seconds to stay alive for debugging once training completes



./gce_scripts/create_instance.sh --instance "$new_instance_name" --image "$boot_image_name" --gce_username "$gce_username" \
  --image-project "$image_project" --machine-type "$machine_type" --maintenance-policy "$maintenance_policy" \
  --min-cpu-platform "$min_cpu_platform" --boot-disk-size "$boot_disk_size" --zone "$zone" \
  --job_id "$job_id" --job_dir "$job_dir" --trainer_module "$trainer_module" \
  --trainer_package_path "$trainer_package_path" --trainer_config "$trainer_config" \
  --train_data_path "$train_data_path" --project "$project" --keep_alive $keep_alive

