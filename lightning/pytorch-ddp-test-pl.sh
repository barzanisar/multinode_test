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


srun launch_training_pl.sh
# module load python # Using Default Python version - Make sure to choose a version that suits your application
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install torchvision pytorch-lightning --no-index

# export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

# # PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# # If it is, it expects the user to have requested one task per GPU.
# # If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!

# srun python pytorch-ddp-test-pl.py  --batch_size 256 --num_workers 8
