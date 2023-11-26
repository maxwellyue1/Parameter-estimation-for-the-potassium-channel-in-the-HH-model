#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=64G
source ENV/bin/activate
python inspect_dataset.py 0 output 10