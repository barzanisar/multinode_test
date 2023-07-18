#!/bin/bash
#SBATCH --nodes 2             
#SBATCH --gres=gpu:t4:4        
#SBATCH --tasks-per-node=1   # number of times to run "srun launch_training_acc.sh" per node, we want to run launch_training_acc.sh once per node
#SBATCH --cpus-per-task=8    # 8 cpus/4gpus per node = 2 cpus per gpu --> set num_workers to 2 in DataLoader
#SBATCH --mem=64000M      
#SBATCH --time=0-01:00       # 1 hr
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-swasland

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_PORT=34567

## Create a virtualenv and install accelerate + its dependencies on all nodes ##
echo "SLURM_NNODES: $SLURM_NNODES"
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env_acc.sh


# srun will run this script <tasks-per-node * nodes> times i.e. 4 x 2 = 8 times
srun launch_training_acc.sh