#!/bin/bash
#SBATCH --nodes 2             
#SBATCH --gres=gpu:4          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=4   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G      
#SBATCH --time=0-01:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-swasland


module load python # Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torchvision --no-index

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

srun python pytorch-ddp-test.py --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS  --batch_size 256 --dist-backend nccl --num_workers 8
