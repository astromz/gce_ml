#!/bin/bash

new_instance_name="test-gpu1"
gce_username="my_gce_username" # account username for your GCE instance
project="my-proj-dev"          # project-id
boot_image_name="gpu1-cpu6-ram30gb-250gb-tensorflow" # pre-created boot image
image_project=$project         # ususally the same as project-id

# VM Instance configurations. Use GCE console's `Create Instance` for reference
machine_type="custom-6-24576"  # 6 CPUs, 24576Mb RAM. Use Cloud Console's Instance Create tool to customize and get your machine_type
maintenance_policy="TERMINATE" # gpu cannot be migrated
accelerator="type=nvidia-tesla-k80,count=1"  # GPU config
min_cpu_platform='Intel Broadwell' # CPU types
boot_disk_size="250"           # 250 Gb
zone="us-east1-c"	       # recommend "us-east1-c"

# Trainer Variables
job_id=$new_instance_name          # unique job id
job_dir="gs://my-proj/test_dir"    # parent dir for your jon on GCS
trainer_module="trainer.task"      # module that actually dose the training
trainer_package_path="./package_example/"  # package on local drive
trainer_config="./trainer_configs/trainer_config.yaml"
train_data_path="gs://my-proj/"    # data need to be pre-uploaded to GCS
keep_alive=500                     # seconds to keep alive for (debugging purpose only) once training completes



./gce_scripts/create_gce_ml_instance.sh --instance "$new_instance_name" --image "$boot_image_name" --gce_username "$gce_username" \
  --image-project "$image_project" --machine-type "$machine_type" --maintenance-policy "$maintenance_policy" \
  --accelerator $accelerator --min-cpu-platform "$min_cpu_platform" --boot-disk-size "$boot_disk_size" \
  --job_id "$job_id" --job_dir "$job_dir" --trainer_module "$trainer_module" \
  --trainer_package_path "$trainer_package_path" --trainer_config "$trainer_config" \
  --train_data_path "$train_data_path" --project "$project" --zone "$zone" --keep_alive $keep_alive
