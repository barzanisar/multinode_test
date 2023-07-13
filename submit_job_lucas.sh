#!/bin/bash
#SBATCH --nodes 2
#SBATCH --gres=gpu:t4:4
#SBATCH --tasks-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:10
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-swasland

export HEAD_NODE=$(hostname) 

srun -N 2 -n 2 config_lucas.sh # set both -N and -n to the number of nodes

srun launch_training_lucas.sh
