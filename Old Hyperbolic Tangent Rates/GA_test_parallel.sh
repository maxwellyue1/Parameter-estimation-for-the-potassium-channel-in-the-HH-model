#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=64G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=30

source ENV/bin/activate
python GA_test_parallel.py 0 output 10 