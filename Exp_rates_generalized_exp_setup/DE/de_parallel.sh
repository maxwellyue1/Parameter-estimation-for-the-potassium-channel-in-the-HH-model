#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=8

python de_parallel.py 0 output 10 