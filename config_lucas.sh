#!/bin/bash

echo "From node ${SLURM_NODEID}: installing virtualenv..."

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index torchvision accelerate

echo "Done installing virtualenv!"

deactivate