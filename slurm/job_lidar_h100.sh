#!/bin/bash

#SBATCH --account=cso@h100
#SBATCH --constraint=h100
#SBATCH --job-name=lidar_only_bs2x32  # Job name
#SBATCH --output=./slurm/h100.log       # Standard output and error log
#SBATCH --error=./slurm/h100.log         # Error log
#SBATCH --nodes=1 # reserve 1 node
#SBATCH --ntasks=2 # reserve 4 tasks (or processes)
#SBATCH --gres=gpu:2              # Request 2 GPUs
##SBATCH --constraint v100-32g
#SBATCH --cpus-per-task=16         # Request 8 CPU cores
#SBATCH --qos=qos_gpu_h100-t3 # QoS
#SBATCH --time=20:00:00           # Time limit (hh:mm:ss)
#SBATCH --mail-user=raphael.sulzer.1@gmail.com  # Email for notifications
#SBATCH --mail-type=ALL           # When to receive emails (BEGIN, END, FAIL, ALL)

#hostname
#nvidia-smi
module purge # purge modules inherited by default
#conda deactivate

# Load modules (if needed)
# module load arch/a100
module load miniforge/24.9.0

# Activate virtual environment (if needed)
conda activate ppp

# recompile the afm module
cd ./pixelspointspolygons/models/hisup/afm_module
make
cd ../../../../

set -x

# Run your Python script

torchrun --nproc_per_node=2 scripts/train.py log_to_wandb=true host=jz run_type=release host.multi_gpu=true dataset=lidarpoly \
experiment_name=lidar_only_bs2x32 checkpoint=validation_best model.batch_size=32 use_lidar=True use_images=False model.fusion=patch_concat
