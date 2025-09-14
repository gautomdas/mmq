#!/bin/bash

#SBATCH --job-name=llava_gqa_fp                    # sets the job name
#SBATCH --output=llava_gqa_fp.%j                    # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=llava_gqa_fp.%j                     # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=02:00:00                                      # how long you would like your job to run; format=hh:mm:ss

#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger                                  # set QOS, this will determine what resources can be requested
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000:1

#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks=1                                              
#SBATCH --ntasks-per-node=1                                      
#SBATCH --mem=32gb                                               # (cpu) memory required by job; if unit is not specified MB will be assumed

module load cuda
source ~/.bashrc
micromamba activate MMQ_LLAVA

python gptq_llava.py --task gqa \
                     --seed 0 \
                     --output_dir /fs/cfar-projects/low-bit-vision/full_precision/gqa_test_do_pad \
                     --no_quant
                    #  --vision-bits 4 \
                    #  --language-bits 4

