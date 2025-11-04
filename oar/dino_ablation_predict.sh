#!/bin/bash 

#OAR -q production 
#OAR -l host=1/gpu=1,walltime=12
#OAR -p gpu-24GB AND gpu_compute_capability_major>=5
#OAR -O oar/runs/dino_ablation_predict.out
#OAR -E oar/runs/dino_ablation_predict.out
#OAR -n dino_ablation_predict

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

python scripts/vit_dino_ablation.py host=g5k host.multi_gpu=false evaluation=test