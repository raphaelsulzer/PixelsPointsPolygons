#!/bin/bash

#SBATCH --account=cso@h100
#SBATCH --job-name=ffl_all_bs4x16  # Job name
#SBATCH --output=./slurm/runs/ffl_all_bs4x16.log       # Standard output and error log
#SBATCH --error=./slurm/runs/ffl_all_bs4x16.log         # Error log
#SBATCH --nodes=1 # reserve 1 node
#SBATCH --ntasks=4 # reserve 4 tasks (or processes)
#SBATCH --gres=gpu:4              # Request 2 GPUs
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16         # Request 8 CPU cores
#SBATCH --qos=qos_gpu_h100-t4 # QoS
#SBATCH --time=48:00:00           # Time limit (hh:mm:ss)
#SBATCH --mail-user=raphael.sulzer.1@gmail.com  # Email for notifications
#SBATCH --mail-type=ALL           # When to receive emails (BEGIN, END, FAIL, ALL)


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

torchrun --nproc_per_node=4 scripts/train.py log_to_wandb=true host=jz run_type=release multi_gpu=true checkpoint=null experiment=ffl_fusion experiment.name=v0_all_bs4x16 country=all