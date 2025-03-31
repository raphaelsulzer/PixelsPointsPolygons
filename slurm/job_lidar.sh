#!/bin/bash

#SBATCH --account=cso@v100
#SBATCH --job-name=lidar_only_bs4x8  # Job name
#SBATCH --output=./slurm/runs/lidar_only.log       # Standard output and error log
#SBATCH --error=./slurm/runs/lidar_only.log         # Error log
#SBATCH --nodes=1 # reserve 1 node
#SBATCH --ntasks=4 # reserve 4 tasks (or processes)
#SBATCH --gres=gpu:4              # Request 2 GPUs
#SBATCH --constraint v100-32g
#SBATCH --cpus-per-task=16         # Request 8 CPU cores
#SBATCH --qos=qos_gpu-t3 # QoS
#SBATCH --time=20:00:00           # Time limit (hh:mm:ss)
#SBATCH --mail-user=raphael.sulzer.1@gmail.com  # Email for notifications
#SBATCH --mail-type=ALL           # When to receive emails (BEGIN, END, FAIL, ALL)


module purge # purge modules inherited by default
#conda deactivate

# Load modules (if needed)
# module load arch/a100
# module load nvidia-compilers
module load cuda/12.1.0
module load miniforge/24.9.0

# Activate virtual environment (if needed)
conda activate ppp2

# recompile the afm module
cd ./pixelspointspolygons/models/hisup/afm_module
make
cd ../../../../

set -x

# Run your Python script

torchrun --nproc_per_node=4 scripts/train.py log_to_wandb=true host=jz run_type=release multi_gpu=true \
experiment_name=lidar_only_bs4x8 checkpoint=null model.batch_size=8 use_lidar=true use_images=false run_type.logging=DEBUG model=hisup

#module load miniforge/24.9.0 && conda activate ppp

#torchrun --nproc_per_node=2 scripts/train.py log_to_wandb=false host=jz run_type=debug multi_gpu=true dataset=lidarpoly experiment_name=debug checkpoint=null model.batch_size=16 use_lidar=true use_images=false run_type=debug log_to_wandb=false

python scripts/train.py log_to_wandb=false host=jz run_type=debug multi_gpu=false dataset=lidarpoly experiment_name=debug checkpoint=null model.batch_size=4 use_lidar=true use_images=false run_type=debug model=hisup