#!/bin/bash

#SBATCH --job-name=blip2_awq_test_caps                    # sets the job name
#SBATCH --output=blip2_awq_test_caps.%j                    # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=blip2_awq_test_caps.%j                     # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=02:00:00                                      # how long you would like your job to run; format=hh:mm:ss

#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger                                  # set QOS, this will determine what resources can be requested
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa5000:1

#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks=1                                              
#SBATCH --ntasks-per-node=1                                      
#SBATCH --mem=128gb                                               # (cpu) memory required by job; if unit is not specified MB will be assumed

module load cuda
source ~/.bashrc
# eval "$(micromamba shell hook --shell bash)"
micromamba activate MMQ

python  ../run_awq.py \
       --config_path captioning_configs/awq_42 \
       --task image_captioning

wait                                                            # wait for any background processes to complete

# once the end of the batch script is reached your job allocation will be revoked
