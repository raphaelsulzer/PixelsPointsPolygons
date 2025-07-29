#!/bin/bash 

#OAR -q production 
#OAR -l host=1/gpu=2,walltime=20
#OAR -p gpu-24GB AND gpu_compute_capability_major>=5
#OAR -O oar/runs/p2p_e_fusion_mnv64.out
#OAR -E oar/runs/p2p_e_fusion_mnv64.out
#OAR -n p2p_e_fusion_mnv64

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

torchrun --nproc_per_node=2 scripts/train.py log_to_wandb=true host=g5k run_type=release host.multi_gpu=true experiment_name=early_fusion_bs2x16_mnv64 checkpoint=null model.batch_size=16 encoder=early_fusion_vit model=pix2poly
