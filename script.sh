#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua40
### -- set the job Name --
#BSUB -J ana_sgai
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 20GB of system-memory
#BSUB -R "rusage[mem=20GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --


export CUDA_HOME=/appl/cuda/12.2.0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/cuda/12.2.0
export HF_HOME=$BLACKHOLE

module load python3/3.10.14

source $BLACKHOLE/social/bin/activate

python3 $BLACKHOLE/SGAI-Final-Project/hpc_graph_making.py > $BLACKHOLE/SGAI-Final-Project/anabanana_job_output.out