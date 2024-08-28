import subprocess
import os

def submit_job(start, end, output_file):
    job_script = f"""#!/bin/bash
#SBATCH --job-name=test_limits
#SBATCH --output={output_file}
#SBATCH --error={output_file}
#SBATCH --array={start}-{end}
#SBATCH --time=00:05:00
#SBATCH --mem=100M
#SBATCH --cpus-per-task=1

echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
"""

    # Write the job script to a temporary file
    script_file = f"job_script_{start}_{end}.sh"
    with open(script_file, "w") as f:
        f.write(job_script)

    # Submit the job
    result = subprocess.run(["sbatch", script_file], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error submitting job {start}-{end}: {result.stderr}")
    else:
        print(f"Submitted job {start}-{end}: {result.stdout.strip()}")

    # Remove the temporary script file
    os.remove(script_file)

def main():
    # Define the ranges for each job array
    ranges = [
        (1, 500),
        (501, 1000),
        (1001, 1500),
        (1501, 2000)
    ]

    # Define a single output file for all jobs
    output_file = "slurm_test_limits_%A_%a.out"

    for start, end in ranges:
        submit_job(start, end, output_file)

if __name__ == "__main__":
    main()
