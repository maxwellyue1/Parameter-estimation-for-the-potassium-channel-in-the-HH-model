#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --mem=84G

source ENV/bin/activate
python train_mlp_updates.py 0 output 10 