#!/bin/bash
#SBATCH --nodes 2                # num nodes
#SBATCH --gres=gpu:t4:4          # Request 4 GPUs = 4 tasks
#SBATCH --tasks-per-node=4       
#SBATCH --cpus-per-task=4       
#SBATCH --mem=64000M      
#SBATCH --time=0-01:00           # 1 hour
#SBATCH --output=%N-%j.out       
#SBATCH --account=rrg-swasland

echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM NTASKS: $SLURM_NTASKS"

srun -N $SLURM_NNODES -n $SLURM_NNODES config_env_pl.sh
wait 10

# srun executes ntasks-per-node * nnodes times i.e. 8 times i.e. once for each GPU
srun launch_training_pl.sh
