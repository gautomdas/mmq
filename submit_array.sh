#!/bin/bash

#SBATCH --job-name=batch_run
#SBATCH --output=output/slurm-%A_%a.out
#SBATCH --error=output/slurm-%A_%a.err
#SBATCH --array=0-499 # Job array size
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --exclude=brigid16,brigid17,brigid18,brigid19,cbcb00,cbcb01,cbcb02,cbcb03,cbcb04,cbcb05,cbcb06,cbcb07,cbcb08,cbcb09,cbcb10,cbcb11,cbcb12,cbcb13,cbcb14,cbcb15,cbcb16,cbcb17,cbcb18,cbcb19,cbcb20,cbcb21,cbcb22,cbcb23,cbcb24,cbcb25,cbcb28,cbcb29,clip00,clip01,clip02,clip03,clip04,clip07,clip08,clip09,clip10,clip11,cml00,cml01,cml02,cml03,cml04,cml05,cml06,cml07,cml08,cml09,cml10,cml11,cml12,cml13,cml14,cml15,cml16,cml17,cml18,cml19,cml20,cml21,cml22,cml23,cml24,cml25,cml26,cml27,cml28,cml31,cml32,cmlcpu00,cmlcpu01,cmlcpu02,cmlcpu03,cmlcpu04,cmlcpu06,cmlcpu07,gammagpu05,legacy00,legacy01,legacy02,legacy03,legacy04,legacy05,legacy06,legacy07,legacy08,legacy09,legacy10,legacy11,legacy13,legacy14,legacy15,legacy16,legacy17,legacy18,legacy19,legacy20,legacy21,legacy22,legacy23,legacy24,legacy25,legacy26,legacy27,legacy28,legacy30,legacygpu00,legacygpu01,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,mbrc00,mbrc01,oasis00,oasis01,oasis02,oasis03,oasis04,oasis05,oasis06,oasis07,oasis08,oasis09,oasis10,oasis11,oasis12,oasis13,oasis14,oasis15,oasis16,oasis17,oasis18,oasis19,oasis20,oasis21,oasis22,oasis23,oasis24,oasis25,oasis26,oasis27,oasis28,oasis29,oasis30,oasis31,oasis32,oasis33,oasis34,oasis35,oasis36,oasis37,oasis38,oasis39,oasis40,quics00,tron06,tron07,tron08,tron09,tron10,tron11,tron12,tron13,tron14,tron15,tron16,tron17,tron18,tron19,tron20,tron21,tron22,tron23,tron24,tron25,tron26,tron27,tron28,tron29,tron30,tron31,tron32,tron33,tron34,tron35,tron36,tron37,tron38,tron39,tron40,tron41,tron42,tron43,tron44,tron62,tron63,tron64,tron65,tron66,tron67,tron68,tron69,twist00,twist01,twist02,twist03,twist04,twist05,vulcan00,vulcan01,vulcan02,vulcan03,vulcan04,vulcan05,vulcan06,vulcan07,vulcan08,vulcan09,vulcan10,vulcan11,vulcan12,vulcan13,vulcan14,vulcan15,vulcan16,vulcan17,vulcan18,vulcan19,vulcan20,vulcan21,vulcan22,vulcan23,vulcan25,vulcan26,vulcan27,vulcan28,vulcan38,vulcan39,vulcan40,vulcan41,vulcan42,vulcan43,vulcan44


# Activate the environment
source ~/.bashrc
micromamba activate blip

# Echo the SLURM array task ID
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Offset: 668"

# Calculate the actual config index based on the task ID
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID + 668))
echo "CONFIG_INDEX: ${CONFIG_INDEX}"

# Run the Python script with the specific config
srun python run.py ./configs/${CONFIG_INDEX}.json
