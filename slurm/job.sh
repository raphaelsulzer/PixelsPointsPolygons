#!/bin/bash
#SBATCH --account=cso@v100
#SBATCH --job-name=p2p_lidarpoly_ori_aug  # Job name
#SBATCH --output=./slurm/output.log       # Standard output and error log
#SBATCH --error=./slurm/error.log         # Error log
#SBATCH --nodes=1 # reserve 1 node
#SBATCH --ntasks=4 # reserve 4 tasks (or processes)
#SBATCH --gres=gpu:4              # Request 2 GPUs
#SBATCH --cpus-per-task=16         # Request 8 CPU cores
#SBATCH --qos=qos_gpu-dev # QoS
#SBATCH --time=16:00:00           # Time limit (hh:mm:ss)
#SBATCH --mail-user=raphael.sulzer.1@gmail.com  # Email for notifications
#SBATCH --mail-type=ALL           # When to receive emails (BEGIN, END, FAIL, ALL)


module purge # purge modules inherited by default
conda deactivate

# Load modules (if needed)
module load arch/a100
module load miniforge/24.9.0

# Activate virtual environment (if needed)
conda activate pix2poly

set -x

# Run your Python script
torchrun --nproc_per_node=4 train.py log_to_wandb=true host=jz run_type=release multi_gpu=true dataset=lidarpoly experiment_name=lidarpoly_ori_aug checkpoint=null model.batch_size=32
