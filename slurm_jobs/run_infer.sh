#!/bin/bash
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/test-infer-%j.out
#SBATCH --error=logs/test-infer-%j.err

# Generic qualitative-inference wrapper. All arguments after the script
# name are forwarded directly to scripts/inference.py.
#
# Examples:
# Geo-MTL, scratch decoder
# sbatch -J infer-geomtl-scratch slurm_jobs/run_infer.sh geomtl --decoder_mode scratch \
#     --checkpoint checkpoints/geomtl/geomtl_scratch_best.pth \
#     --model_weights /path/to/Prithvi_EO_V2_300M.pt --data_path /path/to/data \
#     --output_dir sample_outputs/geomtl_scratch --num_samples 5

# Geo-MTL, pretrained decoder
# sbatch -J infer-geomtl-pretrained slurm_jobs/run_infer.sh geomtl --decoder_mode pretrained \
#     --checkpoint checkpoints/geomtl/geomtl_pretrained_best.pth \
#     --model_weights /path/to/Prithvi_EO_V2_300M.pt --data_path /path/to/data \
#     --output_dir sample_outputs/geomtl_pretrained --num_samples 5

# ResNet-LSTM baseline
# sbatch -J infer-resnet slurm_jobs/run_infer.sh resnet_lstm \
#     --checkpoint checkpoints/baselines/baseline_resnet_lstm.pth \
#     --data_path /path/to/data --output_dir sample_outputs/resnet_lstm

# UNet3D-LSTM baseline
# sbatch -J infer-unet3d-lstm slurm_jobs/run_infer.sh unet3d_lstm \
#     --checkpoint checkpoints/baselines/baseline_unet3d_lstm.pth \
#     --data_path /path/to/data --output_dir sample_outputs/unet3d_lstm

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "Starting Time: $(date)"
echo "Command: python scripts/inference.py $*"
echo "=========================================="

source ~/.bashrc
conda activate prithvi_py311

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python scripts/inference.py "$@"

EXIT_CODE=$?
echo "=========================================="
echo "Inference finished. Exit code: $EXIT_CODE"
echo "Ending Time: $(date)"
echo "=========================================="
exit $EXIT_CODE