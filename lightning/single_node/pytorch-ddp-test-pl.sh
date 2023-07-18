#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:4          
#SBATCH --tasks-per-node=4    
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G      
#SBATCH --time=0-00:10
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-swasland

module load python # Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torchvision pytorch-lightning --no-index

export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

# PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# If it is, it expects the user to have requested one task per GPU.
# If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!

srun python pytorch-ddp-test-pl.py  --batch_size 256 --num_workers 2
