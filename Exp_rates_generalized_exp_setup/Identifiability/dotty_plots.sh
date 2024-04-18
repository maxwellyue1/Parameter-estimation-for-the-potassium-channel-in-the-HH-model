#!/bin/bash
#SBATCH --account=def-awillms
#SBATCH --time=56:00:00
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=8


python dotty_plots.py 0 output 10