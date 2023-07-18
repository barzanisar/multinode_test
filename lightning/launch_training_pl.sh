#!/bin/bash

source $SLURM_TMPDIR/env/bin/activate
export NCCL_BLOCKING_WAIT=1
#export NCCL_ASYNC_ERROR_HANDLING=1

echo "Node $SLURM_NODEID says: Launching python script..."

# $SLURM_NTASKS should be the same as $SLURM_NNODES
echo "SLURM NTASKS: $SLURM_NTASKS"
echo "SLURM_NNODES: $SLURM_NNODES" 

srun python pytorch-ddp-test-pl.py  --batch_size 256 --num_workers 2