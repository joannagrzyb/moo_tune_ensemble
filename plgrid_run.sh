#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH -A plgjoannagrzyb2021a


cd $SLURM_SUBMIT_DIR
pip3 install -r requirements.txt --user
pip3 install -U stream-learn
python3 -W ignore python experiment1_9higher_part1.py