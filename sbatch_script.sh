#!/bin/bash
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16gb
#SBATCH --time=00:10:00

# Load environment
source ~/.bashrc
module load cuda
micromamba activate blip 

# Print Current Compute Node
echo "Running on machine: "
echo $SLURMD_NODENAME

echo "Starting Run Script..."
python run.py ./configs/5.json
