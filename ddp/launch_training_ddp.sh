#!/bin/bash

source $SLURM_TMPDIR/env/bin/activate
export NCCL_BLOCKING_WAIT=1
#export NCCL_ASYNC_ERROR_HANDLING=1

echo "Node $SLURM_NODEID says: main node at $MASTER_ADDR:$MASTER_PORT"
echo "Node $SLURM_NODEID says: Launching python script..."

# $SLURM_NTASKS should be the same as $SLURM_NNODES
echo "SLURM NTASKS: $SLURM_NTASKS"
echo "SLURM_NNODES: $SLURM_NNODES" 

# torch.disstributed.launch is run once per node. It runs the python script on every GPU, and it sets local-rank argument to let the script know which GPU it is using.
# nproc_per_node is set as num GPUs used per node (same as --gres)
# --world_size = total num GPUs over all nodes i.e. 2 nodes with 4 GPUS each so world_size is 2*4=8
# --num_workers per gpu = --cpus_per_task/ num gpus per node (--gres) = 8/4 = 2
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=$SLURM_NNODES --node-rank=$SLURM_NODEID --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT pytorch-ddp-test.py --init_method tcp://$MASTER_ADDR:$MASTER_PORT --world_size 8 --batch_size 256 --dist-backend nccl --num_workers 2
