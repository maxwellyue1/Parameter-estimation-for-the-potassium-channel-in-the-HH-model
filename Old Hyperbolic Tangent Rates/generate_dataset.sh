#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=50G

source ENV/bin/activate
python generate_dataset_main.py 0 output 10 