#!/bin/bash
set -x

module purge # purge modules inherited by default
conda deactivate

# Load modules (if needed)
# module load arch/a100
module load miniforge/24.9.0

# Activate virtual environment (if needed)
conda activate ppp