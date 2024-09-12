#!/bin/bash

#SBATCH --job-name=blip2-vqav2
#SBATCH --output=vqa_baseline.%j
#SBATCH --error=vqa_baseline.%j
#SBATCH --time=20:00:00

#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --acount=vulcan-abhinav
#SBATCH --gres=gpu:p6000:8

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128gb

module load cuda

source ~/.bashrc

micromamba activate blip

python -m torch.distributed.run --nproc_per_node=8 vqav2.py

wait