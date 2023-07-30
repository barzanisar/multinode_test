#!/bin/bash
#SBATCH --nodes 2                # num nodes
#SBATCH --gres=gpu:t4:4          # Request 4 GPUs
#SBATCH --tasks-per-node=1       # number of times to run "srun launch_training_ddp.sh" per node, we want to run launch_training_ddp.sh once per node
#SBATCH --cpus-per-task=8        # number of CPUs per node i.e. each GPU will get 8cpus/4gpus=2cpus, set num_workers in DataLoader to 2
#SBATCH --mem=64000M      
#SBATCH --time=0-01:00           # 1 hour
#SBATCH --output=%N-%j.out       
#SBATCH --account=rrg-swasland

export MASTER_ADDR=$(hostname) # Store the master node’s IP address or name in the MASTER_ADDR environment variable.
export MASTER_PORT=34567       # TCP port of master node

## Create a virtualenv and install your python program dependencies on all nodes ##
# $SLURM_NTASKS should be the same as $SLURM_NNODES
echo "SLURM NTASKS: $SLURM_NTASKS"
echo "SLURM_NNODES: $SLURM_NNODES" 
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env_ddp.sh

# “srun” executes the script <tasks-per-node * nodes> times
srun launch_training_ddp.sh