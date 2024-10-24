import os
from datetime import datetime
import argparse
import shutil
import math
import socket
import itertools
import subprocess

def run(cmd):
    return subprocess.check_output(cmd, shell=True).decode('UTF-8').splitlines()    

def present_in_list(string, gpu_list):
    return any([x in string for x in gpu_list])

def get_exclude_string(gpu_list, default_exclude=None):
    if gpu_list[0] == 'any':
        if default_exclude is None:
            return ''
        else:
            return '#SBATCH --exclude='+','.join(default_exclude)
    memdata = run('sinfo -O nodehost,gres -h')
    superset = set([x.split()[0] for x in memdata])
    blacklist = []
    for x in memdata:
        nodehost, gres = x.strip().split()
        if present_in_list(gres, gpu_list):
            blacklist.append(nodehost)

    exclude_list = superset - set(blacklist)
    if default_exclude:
        exclude_list = exclude_list.union(set(default_exclude))
    exclude_string = ','.join(sorted(exclude_list))
    if exclude_string:
        return '#SBATCH --exclude='+exclude_string+'\n'
    return ''

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=24)
parser.add_argument('--base-dir', default=f'{os.getcwd()}')
parser.add_argument('--output-dirname', default='slurm_files')
parser.add_argument('--partition', default='vulcan', choices=['vulcan','cml','nexus'])
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--qos', default=['scav'], type=str, nargs='*')
parser.add_argument('--env', type=str, required=True, help="Name for this batch of experiments")
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--gpu-type', type=str, default=['p6000'], nargs='*',
                    choices=['any','p6000','gtx','rtx2080','a4000','a5000','a6000'])
parser.add_argument('--cores', default=4, type=int)
parser.add_argument('--mem', default=32, type=int)
parser.add_argument('--filename', default=None, type=str)
parser.add_argument('--max_jobs', default=80, type=int)

args = parser.parse_args()

if args.filename is None:
    args.filename = args.env

# Create output directory
output_dir = os.path.join(args.base_dir, args.output_dirname, args.env)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
print("Output Directory: %s" % output_dir)

# Define parameters for grid search
bit_options = [2, 3, 4, 5, 6, 7, 8, 16]
params = {
    'vision_bits': ['--vision-bits', 'vb', bit_options],
    'qformer_bits': ['--qformer-bits', 'qb', bit_options],
    'language_bits': ['--language-bits', 'lb', [3, 4, 5, 6, 7, 8, 16]]
}

class Argument:
    def __init__(self, name, cmd_line, string_id, val):
        self.name = name
        self.val = val
        self.cmd_string = f' {cmd_line} {str(val)}'
        self.job_string = f'_{string_id}{str(val)}'

# Generate job files
with open(f'{output_dir}/now.txt', "w") as nowfile, \
     open(f'{output_dir}/log.txt', "w") as output_namefile, \
     open(f'{output_dir}/err.txt', "w") as error_namefile, \
     open(f'{output_dir}/name.txt', "w") as namefile:

    arg_list = []
    for key, param in params.items():
        cur_arg_list = []
        for value in param[2]:
            cur_arg_list.append(Argument(key, param[0], param[1], value))
        arg_list.append(cur_arg_list)
    
    n_jobs = 0
    for job_args in itertools.product(*arg_list):
        python_cmd = 'python gptq_blip2.py'
        job_string = ''
        
        for arg in job_args:
            python_cmd += arg.cmd_string
            job_string += arg.job_string
            
        job_string = f'{n_jobs}_{job_string}'
        n_jobs += 1
        
        nowfile.write(f'{python_cmd}\n')
        namefile.write(f'{os.path.join(output_dir, job_string)}.log\n')
        output_namefile.write(f'{os.path.join(output_dir, job_string)}_log.txt\n')
        error_namefile.write(f'{os.path.join(output_dir, job_string)}_error.txt\n')

print(f"\nGenerated {n_jobs} jobs for all bit combinations")

# Create SLURM script
slurm_script_path = os.path.join(output_dir, args.filename + '.slurm')
print(f"Writing SLURM script to {slurm_script_path}")

with open(slurm_script_path, 'w') as slurmfile:
    slurmfile.write("#!/bin/bash\n")
    slurmfile.write(f"#SBATCH --array=1-{n_jobs}%{args.max_jobs}\n")
    slurmfile.write("#SBATCH --output=/dev/null\n")
    slurmfile.write("#SBATCH --error=/dev/null\n")
    slurmfile.write("#SBATCH --requeue\n")
    slurmfile.write("#SBATCH --nodes=1\n")              
    slurmfile.write("#SBATCH --ntasks=1\n")              
    slurmfile.write("#SBATCH --ntasks-per-node=1\n")
    
    # Set up partition and QOS
    if "vulcan" in args.partition:
        slurmfile.write("#SBATCH --account=vulcan\n")
        slurmfile.write("#SBATCH --partition=vulcan-scavenger\n")
        slurmfile.write("#SBATCH --qos=vulcan-scavenger\n")
        slurmfile.write("#SBATCH --nodelist=vulcan[01-07]\n")  # P6000 nodes
    
    # Resource requests
    slurmfile.write(f"#SBATCH --time={args.nhrs}:00:00\n")
    slurmfile.write(f"#SBATCH --cpus-per-task={args.cores}\n")
    slurmfile.write(f"#SBATCH --mem={args.mem}G\n")
    
    # GPU specification
    if args.gpu_type[0] == 'p6000':
        slurmfile.write(f"#SBATCH --gres=gpu:p6000:{args.gpu}\n")
    else:
        slurmfile.write(f"#SBATCH --gres=gpu:{args.gpu}\n")
        slurmfile.write(get_exclude_string(args.gpu_type))
    
    # Environment setup
    slurmfile.write("\nsource ~/.bashrc\n")
    slurmfile.write("module load cuda\n")
    slurmfile.write("micromamba activate blip\n\n")
    
    # Run the job
    slurmfile.write(f'srun --output=$(head -n $SLURM_ARRAY_TASK_ID {output_dir}/log.txt | tail -n 1) $(head -n $SLURM_ARRAY_TASK_ID {output_dir}/now.txt | tail -n 1)\n')

# Submit the job
slurm_command = f"sbatch {slurm_script_path}"
print(f"\nSubmitting job with command: {slurm_command}")
print(f"Total number of jobs: {n_jobs}")
print(f"Running with {args.gpu} GPUs, {args.cores} cores, {args.mem}GB memory for {args.nhrs} hours")

if not args.dryrun:
    os.system(slurm_command)
