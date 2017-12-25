#!/bin/bash

new_instance_name="test-gpu1"
gce_username="207229"
project="nyt-toner-dev"
boot_image_name="gpu1-cpu6-ram30gb-250gb-tensorflow"
image_project=$project

# VM Instance configurations. Use GCE console's `Create Instance` for reference
machine_type="custom-6-24576"
maintenance_policy="TERMINATE" # gpu cannot be migrated
accelerator="type=nvidia-tesla-k80,count=1"  # GPU config 
min_cpu_platform='Intel Broadwell'
boot_disk_size="250"
zone="us-east1-c"

# Trainer Variables
job_id=$new_instance_name
job_dir="gs://nyt-toner/test_dir"
trainer_module="trainer.test"
trainer_package_path="./package/"
trainer_config="./trainer_configs/trainer_config.yaml"
train_data_path="gs://nyt-toner/"  # data need to be pre-uploaded to GCS
project="nyt-toner-dev" 
keep_alive=500



./gce_scripts/create_instance.sh --instance "$new_instance_name" --image "$boot_image_name" --gce_username "$gce_username" \
  --image-project "$image_project" --machine-type "$machine_type" --maintenance-policy "$maintenance_policy" \
  --accelerator $accelerator --min-cpu-platform "$min_cpu_platform" --boot-disk-size "$boot_disk_size" \
  --job_id "$job_id" --job_dir "$job_dir" --trainer_module "$trainer_module" \
  --trainer_package_path "$trainer_package_path" --trainer_config "$trainer_config" \
  --train_data_path "$train_data_path" --project "$project" --zone "$zone" --keep_alive $keep_alive

