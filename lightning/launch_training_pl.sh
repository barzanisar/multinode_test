#!/bin/bash

source $SLURM_TMPDIR/env/bin/activate
export NCCL_BLOCKING_WAIT=1
#export NCCL_ASYNC_ERROR_HANDLING=1

echo "Node $SLURM_NODEID says: Launching python script..."

# run this without srun to run only once per task
python pytorch-pl-test.py  --batch_size 256 --num_workers 4