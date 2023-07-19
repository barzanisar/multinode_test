#!/bin/bash

module load python # Using Default Python version - Make sure to choose a version that suits your application

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torchvision pytorch-lightning --no-index
echo "Done installing virtualenv at $SLURM_TMPDIR/env!"
deactivate