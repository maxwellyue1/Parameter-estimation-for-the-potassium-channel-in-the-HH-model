#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=16G

source ENV/bin/activate
python sensitivity_coeff_matrix.py 0 output 10 