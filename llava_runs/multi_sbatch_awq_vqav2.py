import os
from datetime import datetime
import argparse
import shutil
import math
import time
import socket
import itertools
import subprocess


def run(cmd):
    return subprocess.check_output(cmd, shell=True).decode('UTF-8').splitlines()    

def present_in_list(string, gpu_list):
    return any([x in string for x in gpu_list])

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def get_exclude_string(gpu_list, default_exclude=None):
    if gpu_list[0]  == 'any':
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
        exclude_string = '#SBATCH --exclude='+exclude_string+'\n'
        return exclude_string
    else:
        return ''

def get_include_string(gpu_list, default_include=None):
    if gpu_list[0]  == 'any':
        raise Exception("That's too much, man! (It's a Bojack reference. Watch it if you haven't already, you degenerate)")
    memdata = run('sinfo -O nodehost,gres -h')
    include_list = []
    for x in memdata:
        nodehost, gres = x.strip().split()
        if present_in_list(gres, gpu_list):
            include_list.append(nodehost)
    include_string = ','.join(sorted(include_list))
    if include_string:
        include_string = '#SBATCH --nodelist='+include_string+'\n'
        return include_string
    else:
        return ''
    
# Function to chec for validity of QOS
#TODO: Add time check for QOS

qos_dict = {
            "scav" : {"nhrs" : 72, "cores": 32, "mem":256},
            "high" : {"gpu":4, "cores": 16, "mem":128, "nhrs": 36},
            "medium" : {"gpu":2, "cores": 8, "mem":64, "nhrs": 72},
            "default" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168}}


def check_qos(args):
    
    for qos in args.qos:
        for key, max_value in qos_dict[qos].items():
            val_from_args = getattr(args, key)
            if val_from_args != None:
                if val_from_args > max_value:
                    raise ValueError("Invalid parameter for {} for {}".format(key, qos))
            else:
                setattr(args, key, max_value)
    return args


#TODO: Add day funtionality too 
parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=None)
parser.add_argument('--base-dir', default=f'{os.getcwd()}')
parser.add_argument('--output-dirname', default='outputs')
parser.add_argument('--partition', default='vulcan', choices=['vulcan','cml','nexus'])
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--qos', default=None, type=str, nargs='*', help='Qos to run')
parser.add_argument('--env', type=str, help = "Set the name of the dir you want to dump")
parser.add_argument('--gpu', default=None, type=int, help='Number of gpus')
parser.add_argument('--gpu-type', type=str, help='Type of gpu to use (can be multiple)', default=['any'], 
                    choices=['any','p6000','gtx','rtx2080','a4000','a5000','a6000'], nargs='*')
parser.add_argument('--cores', default=None, type=int, help='Number of cpu cores')
parser.add_argument('--mem', default=None, type=int, help='RAM in G')
parser.add_argument('--single', action='store_true')
parser.add_argument('--filename', default=None, type=str, help='Slurm file name')
parser.add_argument('--max_jobs', default=80, type=int, help='Maximum number of jobs running in parallel')
parser.add_argument('--offset', default=0, type=int, help='Offset')
parser.add_argument('--batchsize', default=500, type=int, help='Offset')

args = parser.parse_args()

if args.filename is None:
    args.filename = args.env

output_dir = os.path.join(args.base_dir, args.output_dirname, args.env)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)

if "nexus" in socket.gethostname():
    root = 'root' ## TODO
else:
    raise Exception("Not on nexus")


# Define parameters for grid search
bit_options = [2,3,4,5,6,8,16]
params = {
    'task': ['--task', 't', ['vqav2']],
    'seed': ['--seed', 's', ['0']],
    'output_dir': ['--output_dir', '', ['/fs/cfar-projects/low-bit-vision/llava/awq/vqav2']],
    'vision_bits': ['--vision-bits', 'vb', bit_options],
    'language_bits': ['--language-bits', 'lb', bit_options]
}
#######################################################################

class Argument(object):

    def __init__(self, name, cmd_line, string_id, val):
        self.name = name
        self.val = val
        if isinstance(val,list):
            if len(val) == 0:

                if isinstance(cmd_line, list):
                    self.cmd_string = ''
                    for cur_line in cmd_line:
                        self.cmd_string += ' '+cur_line+' []'
                else:
                    self.cmd_string = ' '+cmd_line+' []'
            else:
                if isinstance(cmd_line, list):
                    self.cmd_string = ''
                    for cur_line in cmd_line:
                        self.cmd_string += ' '+cur_line+' '+','.join([str(e) for e in val])
                else:
                    self.cmd_string = ' '+cmd_line+' '+','.join([str(e) for e in val])
        else:

            if isinstance(cmd_line, list):
                self.cmd_string = ''
                for cur_line in cmd_line:
                    self.cmd_string += ' '+cur_line+' '+str(val)
            else:
                self.cmd_string = ' '+cmd_line+' '+str(val)
        if isinstance(val,bool):
            if not val:
                self.job_string = ''
                self.cmd_string = ''
                self.name = ''
            else:
                self.job_string = '_'+string_id if string_id else ''
                if isinstance(cmd_line, list):
                    self.cmd_string = ''
                    for cur_line in cmd_line:
                        self.cmd_string += ' '+cur_line+' '
                self.cmd_string = ' '+cmd_line+' '
        elif isinstance(val,list):
            self.job_string = '_'+string_id+'_'.join([str(v) for v in val])
        else:
            self.job_string = '_'+string_id+str(val)
        if string_id == 'none':
            self.job_string = ''

    def copy(self):
        new_arg = Argument(self.name, cmd_line='', string_id='', val=self.val)
        new_arg.cmd_string = self.cmd_string
        new_arg.job_string = self.job_string
        return new_arg
            

os.makedirs(f'{args.base_dir}/{args.output_dirname}/{args.env}',exist_ok=True)
n_jobs = 0
# Making text files which will store the python command to run, stdout, and error if any  
with open(f'{args.base_dir}/{args.output_dirname}/{args.env}/now.txt', "w") as nowfile,\
     open(f'{args.base_dir}/{args.output_dirname}/{args.env}/log.txt', "w") as output_namefile,\
     open(f'{args.base_dir}/{args.output_dirname}/{args.env}/err.txt', "w") as error_namefile,\
     open(f'{args.base_dir}/{args.output_dirname}/{args.env}/name.txt', "w") as namefile:

    arg_list = []
    for key, param in params.items():
        cur_arg_list = []
        if not isinstance(param[2],list):
            param[2] = [param[2]]

        if len(param[2])>1 and key!="dataset":
            assert param[1]!='none', f"{param[0]} set to none with multiple values!"

        for value in param[2]:
            cur_arg_list.append(Argument(key, param[0],param[1], value))

        arg_list.append(cur_arg_list)
    
    arg_list = list(itertools.product(*arg_list))
    n_jobs = 0
    for idx,job_args in enumerate(arg_list):

        # Allows modification of current set of args
        job_args = {arg.name:arg.copy() for arg in job_args}
        
        job_string = ''
        python_cmd = 'python awq_llava.py'
        for arg_name, arg in job_args.items():
            python_cmd += arg.cmd_string
            job_string += arg.job_string

        job_string = f'{n_jobs}_'+job_string
        cmd_line_str = python_cmd
        
        # cmd_line_str = python_cmd

        n_jobs += 1
        
        nowfile.write(f'{cmd_line_str}\n')
        namefile.write(f'{(os.path.join(output_dir, job_string))}.log\n')
        output_namefile.write(f'{(os.path.join(output_dir, job_string))}_log.txt\n')
        error_namefile.write(f'{(os.path.join(output_dir, job_string))}_error.txt\n')
        if args.single:
            break

print(f"\nGenerated {n_jobs} jobs for all bit combinations")
###########################################################################
if len(args.qos)>1:
    splits = split(range(0,n_jobs), len(args.qos))
    for qos in args.qos:
        cur_dir = os.path.join(args.base_dir, args.output_dirname, args.env, qos)
        if os.path.exists(cur_dir):
            shutil.rmtree(cur_dir)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

    with open(f'{args.base_dir}/{args.output_dirname}/{args.env}/log.txt', "r") as output_namefile,\
        open(f'{args.base_dir}/{args.output_dirname}/{args.env}/err.txt', "r") as error_namefile:
        logs = output_namefile.read().splitlines()
        errs = error_namefile.read().splitlines()
    
    with open(f'{args.base_dir}/{args.output_dirname}/{args.env}/log.txt', "w") as output_namefile,\
        open(f'{args.base_dir}/{args.output_dirname}/{args.env}/err.txt', "w") as error_namefile:
        for i,log in enumerate(logs):
            qos_idx = math.floor(i/math.ceil(n_jobs/len(args.qos)))
            folder, basename = os.path.split(log)
            new_log_name = os.path.join(folder, args.qos[qos_idx], basename)
            folder, basename = os.path.split(errs[i])
            new_err_name = os.path.join(folder, args.qos[qos_idx], basename)
            output_namefile.write(f'{new_log_name}\n')
            error_namefile.write(f'{new_err_name}\n')



###########################################################################
#slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
id = args.env.split('run')[-1]
filenames = []
if len(args.qos)==1:
    filenames = [f'{args.qos[0][:2]}_r{id}.slurm' if not args.filename else args.filename]
else:
    for qos in args.qos:
        filenames.append(f'{qos[:2]}_r{id}.slurm' if not args.filename else qos[0]+args.filename)

print("Filenames:")
print(filenames)
slurm_script_paths = [os.path.join(output_dir, filename) for filename in filenames]
slurm_commands = ["sbatch %s" % slurm_script_path for slurm_script_path in slurm_script_paths]
shutil.copyfile(os.path.abspath(__file__),
                os.path.join(output_dir,
                os.path.basename(os.path.abspath(__file__))))


idx = 0
start_idx, end_idx = [], []
for i in range(len(args.qos)):
    start_idx += [idx+1]
    idx += math.ceil(n_jobs/len(args.qos))
    end_idx += [min(idx, n_jobs)]

for i,slurm_script_path in enumerate(slurm_script_paths):
    print(f"writing to {slurm_script_path}")
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        if args.max_jobs>0:
            slurmfile.write(f"#SBATCH --array={start_idx[i]}-{end_idx[i]}%{args.max_jobs}\n")
        else:
            slurmfile.write(f"#SBATCH --array={start_idx[i]}-{end_idx[i]}\n")
        slurmfile.write("#SBATCH --output=/dev/null\n")
        slurmfile.write("#SBATCH --error=/dev/null\n")
        slurmfile.write("#SBATCH --requeue\n")
        args = check_qos(args)

        default_include_list = []
        default_exclude_list = []
        if args.qos[i] == "scav":
            if "vulcan" in args.partition:
                slurmfile.write("#SBATCH --account=vulcan\n")
                slurmfile.write("#SBATCH --partition=vulcan-scavenger\n")
                slurmfile.write("#SBATCH --qos=vulcan-scavenger\n")
                default_exclude_list = ["janus[02-04]"]
            elif "nexus" in args.partition:
                slurmfile.write("#SBATCH --account=scavenger\n")
                slurmfile.write("#SBATCH --partition=scavenger\n")
                slurmfile.write("#SBATCH --qos=scavenger\n")
            elif "cml" in args.partition:
                slurmfile.write("#SBATCH --account=cml-abhinav\n")
                slurmfile.write("#SBATCH --partition=cml-scavenger\n")
                slurmfile.write("#SBATCH --qos=cml-scavenger\n")
        elif args.qos[i] == "high" or args.qos[i] == "medium" or args.qos[i] == "default":
            if "vulcan" in args.partition:
                slurmfile.write("#SBATCH --account=vulcan-abhinav\n")
                slurmfile.write("#SBATCH --partition=vulcan-ampere\n")
                slurmfile.write(f"#SBATCH --qos=vulcan-{args.qos[i]}\n")
                default_exclude_list = ["janus[02-04]"]
            elif "nexus" in args.partition:
                slurmfile.write("#SBATCH --account=nexus\n")
                slurmfile.write(f"#SBATCH --qos={args.qos[i]}\n")
            elif "cml" in args.partition:
                slurmfile.write("#SBATCH --account=cml-abhinav\n")
                slurmfile.write("#SBATCH --partition=cml-dpart\n")
                slurmfile.write(f"#SBATCH --qos=cml-{args.qos[i]}\n")

        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)
        

        if not args.gpu is None: 
            if len(args.gpu_type)==1:
                if 'any' in args.gpu_type:
                    slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
                elif "rtx2080" in args.gpu_type:
                    slurmfile.write("#SBATCH --gres=gpu:rtx2080ti:%d\n" % args.gpu)
                elif "gtx" in args.gpu_type:
                    slurmfile.write("#SBATCH --gres=gpu:gtx1080ti:%d\n" % args.gpu)
                elif "p6000" in args.gpu_type:
                    slurmfile.write("#SBATCH --gres=gpu:p6000:%d\n" % args.gpu)
                elif "a4000" in args.gpu_type:
                    slurmfile.write("#SBATCH --gres=gpu:rtxa4000:%d\n" % args.gpu)
                elif "a5000" in args.gpu_type:
                    slurmfile.write("#SBATCH --gres=gpu:rtxa5000:%d\n" % args.gpu)
                elif "a6000" in args.gpu_type:
                    slurmfile.write("#SBATCH --gres=gpu:rtxa6000:%d\n" % args.gpu)
            else:
                assert len(args.gpu_type)>1
                slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
                # slurmfile.write(get_include_string(args.gpu_type,default_include_list))
                slurmfile.write(get_exclude_string(args.gpu_type,default_exclude_list))
        else:
            raise ValueError("Specify the number of gpus")

        slurmfile.write("\n")
        if "vulcan" in socket.gethostname() or "nexus" in socket.gethostname():
            slurmfile.write(f"cd {root}") #TODO
            # slurmfile.write('conda activate {env}\n') #TODO
            slurmfile.write('source ~/.bashrc')
            slurmfile.write("module load cuda\n")
            slurmfile.write('micromamba activate MMQ_LLAVA\n')
            
        num_exps = 1
        for n in reversed(range(num_exps)):
            slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/{args.output_dirname}/{args.env}/log.txt | tail -n 1)  $(head -n $(expr {num_exps} \* $SLURM_ARRAY_TASK_ID - {n}) {args.base_dir}/{args.output_dirname}/{args.env}/now.txt | tail -n 1)\n")
        slurmfile.write("\n")

for i,slurm_command in enumerate(slurm_commands):
    print(slurm_command)
    print("Running on {}, with {} gpus, {} cores, {} mem for {} hour".format(args.qos[i], args.gpu, args.cores, args.mem , args.nhrs))

if not args.dryrun:
    for slurm_command in slurm_commands:
        os.system("%s &" % slurm_command)
