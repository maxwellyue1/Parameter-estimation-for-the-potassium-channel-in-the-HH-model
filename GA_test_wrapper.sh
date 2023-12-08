#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10

source ENV/bin/activate
python GA_test_wrapper.py 0 output 10 