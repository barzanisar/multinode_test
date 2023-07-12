#!/bin/bash
#SBATCH --nodes 2             
#SBATCH --gres=gpu:t4:4        
#SBATCH --tasks-per-node=4   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000M      
#SBATCH --time=0-01:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-swasland

## Create a virtualenv and install accelerate + its dependencies on all nodes ##
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env_acc.sh

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_PORT=34567

# srun will run this script <tasks-per-node * nodes> times i.e. 4 x 2 = 8 times
srun launch_training_acc.sh