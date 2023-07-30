# Multi-node Multi-GPU training on compute canada

## Data
Download and extract Cifar10 dataset in multinode_test/data folder. You should have multinode_test/data/cifar-10-batches-py folder. 

## Clone
ssh into your account on compute canada and clone this repo

## Using accelerate
1. ssh into your account on compute canada
2. cd to multinode_test/accelerate
3. run `sbatch submit_job.sh`

## Using ddp
1. ssh into your account on compute canada
2. cd to multinode_test/ddp
3. run `sbatch submit_job.sh`

## Using pytorch lightning (not working)
1. ssh into your account on compute canada
2. cd to multinode_test/lightning
3. run `sbatch submit_job.sh`

## Permission denied error

To solve this run `chmod u+x *.sh`

