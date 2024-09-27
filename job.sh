#!/bin/sh
#SBATCH --time=0-01:00:00 #requested time to run the job
#SBATCH --gres=gpu:a100 #allocate a NVIDIA A100 GPU
#SBATCH -c 32   # number of cores
#SBATCH --mem=3952M

python3 training.py
