
# Train .......
✅
source slurm_jobs/local_test_paths.sh
sbatch -J test-geomtl-scratch slurm_jobs/run_train.sh geomtl --decoder_mode scratch \
    --data_path "$DATA_PATH" --model_weights "$MODEL_WEIGHTS" \
    --epochs 1 --checkpoint_dir ./test_checkpoints

✅
sbatch -J test-geomtl-pretrained slurm_jobs/run_train.sh geomtl --decoder_mode pretrained \
    --data_path "$DATA_PATH" --model_weights "$MODEL_WEIGHTS" \
    --epochs 1 --checkpoint_dir ./test_checkpoints

✅
sbatch -J test-resnet-lstm slurm_jobs/run_train.sh resnet_lstm \
    --data_path "$DATA_PATH" --epochs 1 --checkpoint_dir ./test_checkpoints

✅
sbatch -J test-unet3d-lstm slurm_jobs/run_train.sh unet3d_lstm \
    --data_path "$DATA_PATH" --epochs 1 --checkpoint_dir ./test_checkpoints

✅
sbatch -J test-seg-ablation slurm_jobs/run_train.sh seg_ablation \
    --data_path "$DATA_PATH" --model_weights "$MODEL_WEIGHTS" \
    --epochs 1 --checkpoint_dir ./test_checkpoints

👇
Inference:

export DATA_PATH="/home/tadiebirihan.medimem/mtl/multitemporal_crop_data"
export MODEL_WEIGHTS="/home/tadiebirihan.medimem/prithvi_work/Prithvi-EO-2.0-300M/Prithvi_EO_V2_300M.pt"

# Geo-MTL, scratch decoder✅1
sbatch -J infer-geomtl-scratch slurm_jobs/run_infer.sh geomtl --decoder_mode scratch \
    --checkpoint ./test_checkpoints/geomtl_scratch_best.pth \
    --model_weights $MODEL_WEIGHTS --data_path $DATA_PATH \
    --output_dir sample_outputs/geomtl_scratch --num_samples 5

# Geo-MTL, pretrained decoder✅2
sbatch -J infer-geomtl-pretrained slurm_jobs/run_infer.sh geomtl --decoder_mode pretrained \
    --checkpoint ./test_checkpoints/geomtl_pretrained_best.pth \
    --model_weights $MODEL_WEIGHTS --data_path $DATA_PATH \
    --output_dir sample_outputs/geomtl_pretrained --num_samples 5

# ResNet-LSTM baseline✅3
sbatch -J infer-resnet slurm_jobs/run_infer.sh resnet_lstm \
    --checkpoint ./test_checkpoints/baseline_resnet_lstm.pth \
    --data_path $DATA_PATH --output_dir sample_outputs/resnet_lstm

# UNet3D-LSTM baseline✅4
sbatch -J infer-unet3d-lstm slurm_jobs/run_infer.sh unet3d_lstm \
    --checkpoint ./test_checkpoints/baseline_3d_unet_lstm.pth \
    --data_path $DATA_PATH --output_dir sample_outputs/unet3d_lstm

👇
Eval:
export DATA_PATH="/home/tadiebirihan.medimem/mtl/multitemporal_crop_data"
export MODEL_WEIGHTS="/home/tadiebirihan.medimem/prithvi_work/Prithvi-EO-2.0-300M/Prithvi_EO_V2_300M.pt"

# Geo-MTL, scratch decoder✅ 1
sbatch -J eval-geomtl-scratch slurm_jobs/run_eval.sh geomtl --decoder_mode scratch \
    --checkpoint ./test_checkpoints/geomtl_scratch_best.pth \
    --model_weights $MODEL_WEIGHTS --data_path $DATA_PATH \
    --output_file results/metrics_geomtl_scratch.json

# Geo-MTL, pretrained decoder✅ 2
sbatch -J eval-geomtl-pretrained slurm_jobs/run_eval.sh geomtl --decoder_mode pretrained \
    --checkpoint ./test_checkpoints/geomtl_pretrained_best.pth \
    --model_weights $MODEL_WEIGHTS --data_path $DATA_PATH \
    --output_file results/metrics_geomtl_pretrained.json --fast

# ResNet-LSTM baseline ✅ 3
sbatch -J eval-resnet slurm_jobs/run_eval.sh resnet_lstm \
    --checkpoint ./test_checkpoints/baseline_resnet_lstm.pth \
    --data_path $DATA_PATH --output_file results/metrics_resnet_lstm.json

# UNet3D-LSTM baseline ✅ 4
sbatch -J eval-unet3d-lstm slurm_jobs/run_eval.sh unet3d_lstm \
    --checkpoint ./test_checkpoints/baseline_3d_unet_lstm.pth \
    --data_path $DATA_PATH --output_file results/metrics_unet3d_lstm.json

# Segmentation ablation ✅ 5
sbatch -J eval-seg slurm_jobs/run_eval.sh seg_ablation \
    --checkpoint ./test_checkpoints/seg_ablation_best.pth \
    --data_path $DATA_PATH --output_file results/metrics_seg_ablation.json



==================================
# import test.......
source ~/.bashrc
conda activate prithvi_py311
cd /home/tadiebirihan.medimem/geoMTL/igarss
python -c "from models.geomtl import GeoMTL; from models.baselines import ResNetLSTM; from models.seg_ablation import SegOnlyGeoAgent; print('imports OK')"

— `source` exports the variables into your current terminal session/only once/, so as long as you don't close the terminal/log out, `$DATA_PATH` and `$MODEL_WEIGHTS` stay set for every subsequent `sbatch` command; you'd only need to re-run it after opening a new SSH session or shell.

(--fast skips METEOR — good for a quick sanity check, since a 1-epoch checkpoint will produce weak/degenerate captions that could hang METEOR)