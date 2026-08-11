#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Generic training wrapper. All arguments after the script name are
# forwarded directly to scripts/train.py.
#
# Examples:
#   sbatch -J geomtl-scratch slurm_jobs/run_train.sh geomtl --decoder_mode scratch \
#       --data_path /path/to/data --model_weights /path/to/Prithvi_EO_V2_300M.pt
#
#   sbatch -J geomtl-pretrained slurm_jobs/run_train.sh geomtl --decoder_mode pretrained \
#       --itc_weight 1.0 --data_path /path/to/data --model_weights /path/to/Prithvi_EO_V2_300M.pt
#
#   sbatch -J resnet-lstm slurm_jobs/run_train.sh resnet_lstm --data_path /path/to/data
#
#   sbatch -J unet3d-lstm slurm_jobs/run_train.sh unet3d_lstm --data_path /path/to/data
#
#   sbatch -J seg-ablation slurm_jobs/run_train.sh seg_ablation --finetune_full_model \
#       --data_path /path/to/data --model_weights /path/to/Prithvi_EO_V2_300M.pt
#
# Override resource requests at submission time if needed, e.g.:
#   sbatch --time=1-00:00:00 --mem=64G ...

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "Starting Time: $(date)"
echo "Command: python scripts/train.py $*"
echo "=========================================="

source ~/.bashrc
conda activate prithvi_py311  # must match the environment name in environment.yml

# Repo root is the parent of this slurm_jobs/ directory -- no hardcoded paths.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python scripts/train.py "$@"

EXIT_CODE=$?
echo "=========================================="
echo "Training finished. Exit code: $EXIT_CODE"
echo "Ending Time: $(date)"
echo "=========================================="
exit $EXIT_CODE