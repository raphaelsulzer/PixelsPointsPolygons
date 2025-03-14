# OAR cheetsheat
https://oar.imag.fr/docs/2.5/user/quickstart.html



## to submit run 
oarsub -S ./oar/job.sh
## to see status run
oarstat -fj $JOBID
## kill a job
oardel $JOBID
