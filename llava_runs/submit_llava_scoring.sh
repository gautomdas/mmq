#!/bin/bash

#SBATCH --job-name=llava_scoring                    # sets the job name
#SBATCH --output=llava_scoring.%j                    # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=llava_scoring.%j                     # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=03:00:00                                      # how long you would like your job to run; format=hh:mm:ss

#SBATCH --partition=tron
#SBATCH --qos=high                                  # set QOS, this will determine what resources can be requested
#SBATCH --account=nexus


#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks=1                                              
#SBATCH --ntasks-per-node=1                                      
#SBATCH --mem=32gb                                               # (cpu) memory required by job; if unit is not specified MB will be assumed

source ~/.bashrc
micromamba activate MMQ_LLAVA

srun python llava_scoring.py
