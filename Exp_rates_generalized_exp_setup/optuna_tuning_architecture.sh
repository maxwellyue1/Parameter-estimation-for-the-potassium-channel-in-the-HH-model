#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G
#SBATCH --nodes=1   
#SBATCH --time=26:00:00

source ../ENV/bin/activate
python optuna_tuning_architecture.py 0 output 10 