#!/bin/bash
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/test-eval-%j.out
#SBATCH --error=logs/test-eval-%j.err

# Generic evaluation wrapper. All arguments after the script name are
# forwarded directly to scripts/evaluate.py.
#
# Examples:
# Geo-MTL, scratch decoder
# sbatch -J eval-geomtl-scratch slurm_jobs/run_eval.sh geomtl --decoder_mode scratch \
#     --checkpoint checkpoints/geomtl/geomtl_scratch_best.pth \
#     --model_weights /path/to/Prithvi_EO_V2_300M.pt --data_path /path/to/data \
#     --output_file results/metrics_geomtl_scratch.json

# # Geo-MTL, pretrained decoder
# sbatch -J eval-geomtl-pretrained slurm_jobs/run_eval.sh geomtl --decoder_mode pretrained \
#     --checkpoint checkpoints/geomtl/geomtl_pretrained_best.pth \
#     --model_weights /path/to/Prithvi_EO_V2_300M.pt --data_path /path/to/data \
#     --output_file results/metrics_geomtl_pretrained.json --fast

# # ResNet-LSTM baseline
# sbatch -J eval-resnet slurm_jobs/run_eval.sh resnet_lstm \
#     --checkpoint checkpoints/baselines/baseline_resnet_lstm.pth \
#     --data_path /path/to/data --output_file results/metrics_resnet_lstm.json

# # UNet3D-LSTM baseline
# sbatch -J eval-unet3d-lstm slurm_jobs/run_eval.sh unet3d_lstm \
#     --checkpoint checkpoints/baselines/baseline_unet3d_lstm.pth \
#     --data_path /path/to/data --output_file results/metrics_unet3d_lstm.json

# # Segmentation ablation
# sbatch -J eval-seg slurm_jobs/run_eval.sh seg_ablation \
#     --checkpoint checkpoints/seg_ablation/seg_ablation_best.pth \
#     --data_path /path/to/data --output_file results/metrics_seg_ablation.json

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "Starting Time: $(date)"
echo "Command: python scripts/evaluate.py $*"
echo "=========================================="

source ~/.bashrc
conda activate prithvi_py311
# pip install -q pycocoevalcap  # first-run only; skip if already installed

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python scripts/evaluate.py "$@"

EXIT_CODE=$?
echo "=========================================="
echo "Evaluation finished. Exit code: $EXIT_CODE"
echo "Ending Time: $(date)"
echo "=========================================="
exit $EXIT_CODE