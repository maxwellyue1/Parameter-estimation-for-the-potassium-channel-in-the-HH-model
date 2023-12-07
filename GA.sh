#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=12G
#SBATCH --ntasks-per-node=4

source ENV/bin/activate
python GA_paralell.py 0 output 10 