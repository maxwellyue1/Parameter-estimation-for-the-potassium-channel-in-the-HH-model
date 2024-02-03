#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=50G

source ENV/bin/activate
python generate_data_main_varying_setup.py 0 output 10 