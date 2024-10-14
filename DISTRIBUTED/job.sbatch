#!/bin/bash -l
#SBATCH -N 2			#(=num_nodes)
#SBATCH --ntasks-per-node=2	#(=devices)
#SBATCH --time=0-01:00:00 #requested time to run the job
#SBATCH --gres=gpu:a100:2 #allocate 2 NVIDIA A100 GPU
#SBATCH -c 32   # number of cores
#SBATCH --mem=8G

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

source ../mypython/bin/activate

python3 tokenize_squad.py

srun python3 lightning_training.py
