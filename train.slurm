#!/bin/bash
#SBATCH --job-name=halfkp-train
#SBATCH --output=logs/halfkp-train-%j.log
#SBATCH --error=logs/halfkp-train-%j.err
#SBATCH --time=24:00:00
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

python -u marlinflow/trainer/main.py --data-root data/ --lr 0.001 --epochs 10 --scale 400 --wandb-project marlinflow --wandb-entity torvim

echo "Training job completed at $(date)"
