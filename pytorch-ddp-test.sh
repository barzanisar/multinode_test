#!/bin/bash
#SBATCH --nodes 8             
#SBATCH --gres=gpu:t4:4          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=1   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M      
#SBATCH --time=0-01:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-swasland

#export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_PORT=34567

## Create a virtualenv and install accelerate + its dependencies on all nodes ##
echo "SLURM_NNODES: $SLURM_NNODES"
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env_ddp.sh

srun launch_training_ddp.sh