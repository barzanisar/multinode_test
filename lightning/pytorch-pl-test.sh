#!/bin/bash
#SBATCH --nodes 2                # num nodes
#SBATCH --gres=gpu:t4:4          # Request 4 GPUs
#SBATCH --tasks-per-node=4       
#SBATCH --cpus-per-task=2        
#SBATCH --mem=64000M      
#SBATCH --time=0-01:00           # 1 hour
#SBATCH --output=%N-%j.out       
#SBATCH --account=rrg-swasland

echo "SLURM_NNODES: $SLURM_NNODES"
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env_pl.sh
wait 10

srun launch_training_pl.sh
