#!/bin/bash

#SBATCH --job-name=llava_vqav2_v4_l2                    # sets the job name
#SBATCH --output=llava_vqav2_v4_l2.%j                    # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=llava_vqav2_v4_l2.%j                     # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=3:00:00                                      # how long you would like your job to run; format=hh:mm:ss

#SBATCH --partition=tron
#SBATCH --qos=high                               # set QOS, this will determine what resources can be requested
#SBATCH --account=nexus
#SBATCH --gres=gpu:rtxa5000:1

#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks=1                                              
#SBATCH --ntasks-per-node=1                                      
#SBATCH --mem=32gb                                               # (cpu) memory required by job; if unit is not specified MB will be assumed

module load cuda
source ~/.bashrc
micromamba activate MMQ_LLAVA

python gptq_llava.py --task vqav2 \
                     --seed 42 \
                     --output_dir /fs/cfar-projects/low-bit-vision/llava/gptq/vqav2_subset \
                     --vision-bits 4 \
                     --language-bits 2