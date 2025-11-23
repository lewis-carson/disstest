#!/bin/bash

# 1. Pull latest changes
echo "Pulling latest changes..."
git pull

# 2. Get the running process ID for user dhqq26
# -h: no header
# -o %A: output only the job ID
JOB_IDS=$(squeue -u dhqq26 -h -o %A)

# 3. Cancel the job(s) if found
if [ -n "$JOB_IDS" ]; then
    echo "Found running job(s): $JOB_IDS"
    for id in $JOB_IDS; do
        echo "Cancelling job $id..."
        scancel "$id"
    done
else
    echo "No running jobs found for user dhqq26."
fi

# 4. Submit the new job
# The user wrote "squeue train.slurm" but likely meant "sbatch train.slurm" to start it.
echo "Submitting train.slurm..."
sbatch train.slurm

# 5. Check the queue
echo "Checking queue..."
squeue -u dhqq26
