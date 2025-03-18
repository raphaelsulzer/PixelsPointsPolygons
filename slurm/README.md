# Jean-Zay doc
http://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf

## cheatsheet
http://www.idris.fr/media/su/idrischeatsheet.pdf

## Show job queue for user.
squeue -lu $USER
## Show pending jobs and the estimated start time for user idrisid
squeue -u $USER --start
## Display control states for a job
scontrol show job jobid 
## Display control states for a node
scontrol show node nodeid 
scancel jobid Delete the job with jobid
sinfo Display the system settings
sacct Display job accounting information
sbatch filename Submit a batch script filename
salloc Obtain an interactive job allocation
srun execfile Obtain a job allocation and execute
an application