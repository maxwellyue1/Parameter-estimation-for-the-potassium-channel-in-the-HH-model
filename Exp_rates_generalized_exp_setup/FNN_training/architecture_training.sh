#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --mem=84G
#SBATCH --nodes=1   
#SBATCH --time=24:00:00

module python
source ../../ENV/bin/activate
python architecture_training.py 0 output 10 