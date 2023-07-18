#!/bin/bash

source $SLURM_TMPDIR/env/bin/activate
export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO

echo "Node $SLURM_NODEID says: main node at $MASTER_ADDR:$MASTER_PORT"
echo "Node $SLURM_NODEID says: Launching python script with accelerate..."
echo "SLURM NTASKS: $SLURM_NTASKS"
echo "SLURM_NNODES: $SLURM_NNODES"

accelerate launch \
--multi_gpu \
--gpu_ids="all" \
--num_machines=$SLURM_NNODES \
--machine_rank=$SLURM_NODEID \
--num_processes=8 \ # This is the total number of GPUs across all nodes
--main_process_ip="$MASTER_ADDR" \
--main_process_port=$MASTER_PORT \
pytorch-acc-test.py --batch_size 256 --num_workers=2

