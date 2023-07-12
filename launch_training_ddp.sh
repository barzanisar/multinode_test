#!/bin/bash

source $SLURM_TMPDIR/env/bin/activate
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Node $SLURM_NODEID says: main node at $MASTER_ADDR"
echo "Node $SLURM_NODEID says: Launching python script with accelerate..."
echo "SLURM NTASKS: $SLURM_NTASKS"
echo "SLURM_NNODES: $SLURM_NNODES"


# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
#srun python pytorch-ddp-test.py --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS  --batch_size 256 --dist-backend nccl --num_workers 4
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=$SLURM_NNODES --node-rank=$SLURM_NODEID --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT pytorch-ddp-test.py.py --init_method tcp://$MASTER_ADDR:34567 --world_size $SLURM_NTASKS  --batch_size 256 --dist-backend nccl --num_workers 4
