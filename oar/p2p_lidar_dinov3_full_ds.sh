#!/bin/bash 

#OAR -q production 
#OAR -l host=1/gpu=2,walltime=24
#OAR -p gpu-24GB AND gpu_compute_capability_major>=5
#OAR -O oar/runs/p2p_lidar_dinov3_full_ds.out
#OAR -E oar/runs/p2p_lidar_dinov3_full_ds.out
#OAR -n p2p_lidar_dinov3_full_ds

# display some information about attributed resources
hostname 
nvidia-smi 

# make use of a python torch environment
module load conda
conda activate p3pt2.9
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.6+PTX"
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))";
pip install -e .

# to get rid of this annoying warning: sh: /home/rsulzer/.conda/envs/ppp/bin/../lib/libtinfo.so.6: no version information available (required by sh)
export LD_BIND_NOW=1
# recompile the afm module
cd ./pixelspointspolygons/models/hisup/afm_module
make
cd ../../../../

torchrun --nproc_per_node=2 --master-port=$((10000 + RANDOM % 50000)) scripts/train.py run_type=release host=g5k experiment=p2p_lidar_dinov3 experiment.name=p2p_lidar_dinov3_full_ds experiment.group_name=v3_pix2poly experiment.dataset.country=all
