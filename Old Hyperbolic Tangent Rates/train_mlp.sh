#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --mem=64G

mkdir -p train_output
source ENV/bin/activate
python train_mlp.py 0 output 10 