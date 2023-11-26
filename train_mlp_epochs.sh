#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --mem=86G

mkdir -p train_output
source ENV/bin/activate
python train_mlp_epochs.py 0 output 10 