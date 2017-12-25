#!/bin/bash

new_instance_name="test-cpu1"
gce_username='207229'
project="nyt-toner-dev"
boot_image_name="test-image-2cpu-6gb"
image_project=$project

# VM Instance configurations. Use GCE console's `Create Instance` for reference
machine_type="custom-2-5120"
maintenance_policy="MIGRATE" # MIGRATE for cpu; TERMINATE for gpu
min_cpu_platform='Intel Broadwell'
boot_disk_size="50"
zone="us-east1-c"


# Trainer Variables
job_id="testjob_cpu"
job_dir="gs://nyt-toner/test_dir"
trainer_module="trainer.test"
trainer_package_path="./package/"
trainer_config="./trainer_configs/trainer_config.yaml"
train_data_path="gs://nyt-toner/"  # data need to be pre-uploaded to GCS
project="nyt-toner-dev" 
keep_alive=700



./gce_scripts/create_instance.sh --instance "$new_instance_name" --image "$boot_image_name" --gce_username "$gce_username" \
  --image-project "$image_project" --machine-type "$machine_type" --maintenance-policy "$maintenance_policy" \
  --min-cpu-platform "$min_cpu_platform" --boot-disk-size "$boot_disk_size" --zone "$zone" \
  --job_id "$job_id" --job_dir "$job_dir" --trainer_module "$trainer_module" \
  --trainer_package_path "$trainer_package_path" --trainer_config "$trainer_config" \
  --train_data_path "$train_data_path" --project "$project" --keep_alive $keep_alive

