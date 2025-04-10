#!/bin/bash 

#OAR -q production 
#OAR -l host=1/gpu=2,walltime=16
#OAR -p gpu-24GB AND gpu_compute_capability_major>=5
#OAR -O oar/runs/image_hisup_vit_cnn.out
#OAR -E oar/runs/image_hisup_vit_cnn.out

# display some information about attributed resources
hostname 
nvidia-smi 

# make use of a python torch environment
module load conda
conda activate ppp
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))";

# to get rid of this annoying warning: sh: /home/rsulzer/.conda/envs/ppp/bin/../lib/libtinfo.so.6: no version information available (required by sh)
export LD_BIND_NOW=1
# recompile the afm module
cd ./pixelspointspolygons/models/hisup/afm_module
make
cd ../../../../

torchrun --nproc_per_node=2 scripts/train.py log_to_wandb=true host=g5k run_type=release multi_gpu=true experiment_name=image_vit_cnn_bs2x16 checkpoint=null model.batch_size=16 encoder=vit_cnn

# python scripts/train.py log_to_wandb=false host=g5k run_type=debug multi_gpu=false experiment_name=debug checkpoint=null model.batch_size=16 use_lidar=False use_images=True

# git pull && torchrun --nproc_per_node=2 scripts/train.py log_to_wandb=false host=g5k run_type=debug multi_gpu=true experiment_name=debug checkpoint=null model.batch_size=2 use_lidar=False use_images=True update_pbar_every=0.5

# python scripts/train.py log_to_wandb=false host=g5k run_type=release multi_gpu=false experiment_name=debug checkpoint=null model.batch_size=4 encoder=vit_cnn update_pbar_every=0.5