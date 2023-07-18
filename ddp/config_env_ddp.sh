#!/bin/bash

module load python

virtualenv --no-download $SLURM_TMPDIR/env

source $SLURM_TMPDIR/env/bin/activate

pip install --upgrade pip --no-index

pip install --no-index torchvision

echo "Done installing virtualenv at $SLURM_TMPDIR/env!"