#!/bin/bash 

#OAR -q production 
#OAR -l host=1/gpu=2,walltime=16
#OAR -p gpu-24GB AND gpu_compute_capability_major>=5
#OAR -O oar/runs/%jobid%.out
#OAR -E oar/runs/%jobid%.err 

# display some information about attributed resources
hostname 
nvidia-smi 

# make use of a python torch environment
module load conda
conda activate ppp
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))";

# recompile the afm module
cd ./pixelspointspolygons/models/hisup/afm_module
make
cd ../../../../

python scripts/train.py log_to_wandb=true host=g5k run_type=release multi_gpu=false dataset=lidarpoly experiment_name=lidarpoly_D4_aug_bs32 checkpoint=null model.batch_size=32

