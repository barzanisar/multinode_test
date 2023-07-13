#!/bin/bash

source $SLURM_TMPDIR/env/bin/activate
export NCCL_BLOCKING_WAIT=1
#export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_DEBUG=INFO

echo "r$SLURM_NODEID master: $HEAD_NODE"
echo "r$SLURM_NODEID Launching python script"

time accelerate launch \
--multi_gpu \
--gpu_ids="all" \
--num_machines=$SLURM_NNODES \
--machine_rank=$SLURM_NODEID \
--num_processes=4 \
--main_process_ip="$HEAD_NODE" \
--main_process_port=34567 \
pytorch-lucas.py --batch_size 4096 --num_workers=2

