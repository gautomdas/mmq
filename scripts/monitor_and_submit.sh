#!/bin/bash

# Function to check if user has any jobs in the queue
user=""
check_queue() {
    squeue -u $user | grep -q $user 
    return $?
}

# Function to submit next batch of jobs
submit_next_batch() {
    python submit_jobs.py --offset 1000 --batchsize 500 --output_dir output
}

# Main loop
while true; do
    if ! check_queue; then
        echo "No jobs in queue. Submitting next batch..."
        submit_next_batch
        echo "Next batch submitted. Waiting for 5 minutes before checking again..."
        sleep 300  # Wait for 5 minutes
    else
        echo "Jobs still running. Checking again in 1 minute..."
        sleep 60  # Wait for 1 minute
    fi
done
