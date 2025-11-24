#!/bin/bash
#SBATCH --job-name=halfkp-train
#SBATCH --output=logs/halfkp-train-%j.log
#SBATCH --error=logs/halfkp-train-%j.err
#SBATCH --time=1:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:1g.10gb:1


echo "Starting HalfKP training job"
echo "Data directory: $DATA_DIR"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi -L)"

source .venv/bin/activate

python -u trainer/main.py \
	--data-root data/ \
	--lr 0.001 \
	--epochs 5 \
	--batch-size 16384 \
	--scale 400 \
	--replay-buffer-size 64 \
	--replay-prob 0.25 \
	--replay-min-batches 4 \
	--replay-priority-metric cp_abs_mean \
	--replay-priority-eps 1e-6 \
    --wandb-project marlinflow \
    --wandb-entity torvim

echo "Training job completed at $(date)"
