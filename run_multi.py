import os
import time

# Define the total number of configurations and the SLURM array size limit
total_configs = 18
max_array_size = 4
delay_seconds = 3  # Number of seconds to wait between submissions

# Calculate the number of batches needed
num_batches = (total_configs + max_array_size - 1) // max_array_size

# Set the batch size for each command execution
batch_size = max_array_size

for batch in range(num_batches):
    start_offset = batch * max_array_size
    batch_size = min(max_array_size, total_configs - start_offset)

    # Construct the command with the batch-specific offset and batch size
    command = f"python multi_sbatch.py --env trial \
                                       --nhrs 8 \
                                       --qos scav \
                                       --partition nexus \
                                       --gpu 8 --gpu-type a5000 a6000 \
                                       --filename SLURM_{start_offset} \
                                       --cores 1 \
                                       --mem 128 \
                                       --base-dir ./ \
                                       --output-dirname slurm_files \
                                       --offset {start_offset} \
                                       --batchsize {batch_size}"

    # Execute the command
    os.system(command)

    # Wait for a few seconds before the next submission
    time.sleep(delay_seconds)

