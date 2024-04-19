#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=50G
#SBATCH --time=24:00:00

python generate_data_main_varying_setup.py 0 output 10 