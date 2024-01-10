#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=32G

source ENV/bin/activate
python sensitivity_coeff_matrix.py 0 output 10 