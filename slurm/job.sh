#!/bin/bash
#SBATCH -A cso@a100
#SBATCH --job-name=p2p_lidarpoly_ori_aug  # Job name
#SBATCH --output=./slurm/output.log       # Standard output and error log
#SBATCH --error=./slurm/error.log         # Error log
#SBATCH --partition=gpu           # GPU partition
#SBATCH --gres=gpu:2              # Request 2 GPUs
#SBATCH --cpus-per-task=16         # Request 8 CPU cores
#SBATCH --mem=32G                 # Request 32GB RAM
#SBATCH --time=16:00:00           # Time limit (hh:mm:ss)
#SBATCH --mail-user=raphael.sulzer.1@gmail.com  # Email for notifications
#SBATCH --mail-type=ALL           # When to receive emails (BEGIN, END, FAIL, ALL)

# Load modules (if needed)
module load miniforge/24.9.0

# Activate virtual environment (if needed)
conda activate pix2poly

# Run your Python script
torchrun --nproc_per_node=2 train.py log_to_wandb=true host=jz run_type=release multi_gpu=true dataset=lidarpoly experiment_name=lidarpoly_ori_aug checkpoint=null
