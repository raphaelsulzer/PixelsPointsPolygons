# Jean-Zay doc
http://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf

## cheatsheet
http://www.idris.fr/media/su/idrischeatsheet.pdf

## qos per gpu type
http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm-eng.html#for_a100_partition

## Show job queue for user.
squeue -lu $USER
## Show pending jobs and the estimated start time for user idrisid
squeue -u $USER --start
## Display control states for a job
scontrol show job jobid 
## Display control states for a node
scontrol show node nodeid 
## Delete the job with jobid
scancel jobid 
## Display the system settings
sinfo 
## Display job accounting information
sacct 
##  Submit a batch script filename
sbatch filename
## Obtain an interactive job allocation
salloc 
## Obtain a job allocation and execute an application
srun execfile 

## Indicate the CPU and/or GPU hours allocations
idracct 

## activate conda evn
module load miniforge/24.9.0 && conda activate ppp

### reserve interactive node
srun --ntasks=2 --gres=gpu:2 --account=cso@v100 --time=01:00:00 --pty bash

srun --ntasks=2 --gres=gpu:2 --account=cso@a100 --qos=qos_gpu_a100-t3 --constraint=a100 --time=01:00:00 --pty bash

srun --ntasks=2 --gres=gpu:2 --account=cso@h100 --qos=qos_gpu_h100-t4 --constraint=h100 --time=00:30:00 --pty bash