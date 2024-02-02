#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --mem=10G
source ../ENV/bin/activate
python read_num_samples.py 0 output 10
