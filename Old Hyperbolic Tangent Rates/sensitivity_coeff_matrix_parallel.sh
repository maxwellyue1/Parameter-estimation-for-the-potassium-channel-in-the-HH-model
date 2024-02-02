#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --ntasks-per-node=50
#SBATCH --mem=16G

source ENV/bin/activate
python sensitivity_coeff_matrix_parallel.py 0 output 10 