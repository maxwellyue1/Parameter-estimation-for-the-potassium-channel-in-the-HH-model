#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --mem=64G
#SBATCH --nodes=1   
#SBATCH --time=26:00:00

source ../ENV/bin/activate
python mlp_training.py 0 output 10 