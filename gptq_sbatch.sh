#!/bin/bash
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:p6000:1
#SBATCH --nodelist=vulcan[01-07]     # excluding vulcan00 since it has a mixed setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --job-name=blip_gptq
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Load environment
source ~/.bashrc
module load cuda
micromamba activate blip

# Print Current Compute Node
echo "Running on machine: "
echo $SLURMD_NODENAME
echo "Starting Run Script..."

# Run line
python gptq_blip2.py --vision-bits 4 --qformer-bits 5 --language-bits 4
