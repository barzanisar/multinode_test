# Multi-node Multi-GPU training on compute canada

## Data
Download and extract Cifar10 dataset in multinode_test/data folder. You should have multinode_test/data/cifar-10-batches-py folder.

## Using accelerate
1. ssh into your account on compute canada
2. cd to multinode_test/accelerate
3. run `sbatch pytorch-acc-test.sh`

## Using ddp
1. ssh into your account on compute canada
2. cd to multinode_test/ddp
3. run `sbatch pytorch-ddp-test.sh`

## Using pytorch lightning
1. ssh into your account on compute canada
2. cd to multinode_test/lightning
3. run `sbatch pytorch-pl-test.sh`

## Permission denied error

To solve this run `chmod u+x *.sh`

