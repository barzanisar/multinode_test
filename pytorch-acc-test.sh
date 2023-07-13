#!/bin/bash
#SBATCH --nodes 2             
#SBATCH --gres=gpu:t4:4        
#SBATCH --tasks-per-node=1   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M      
#SBATCH --time=0-00:10
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-swasland

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
#export MASTER_PORT=34567

## Create a virtualenv and install accelerate + its dependencies on all nodes ##
echo "SLURM_NNODES: $SLURM_NNODES"
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env_acc.sh


# srun will run this script <tasks-per-node * nodes> times i.e. 4 x 2 = 8 times
srun launch_training_acc.sh