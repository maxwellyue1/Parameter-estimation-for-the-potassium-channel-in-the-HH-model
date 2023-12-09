#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10

source ENV/bin/activate
python GA_paralell.py 0 output 10 