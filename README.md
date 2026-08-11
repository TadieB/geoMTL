## Dataset

Due to size constraints, the dataset and checkpoints are hosted externally.

1. Download the [Multi-Temporal Crop Classification dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification/tree/main) and the corresponding [GeoMTL text annotations](https://huggingface.co/datasets/trust-tad/geomtl-dataset/tree/main).
2. Extract both into the same directory. It should contain `chips_df.csv`, `training_data.txt`, `validation_data.txt`, `test_data.txt`, `training_chips/`, `validation_chips/`, and the per-chip `.txt` / `.qual.txt` / `.clean.txt` caption files.
3. Pretrained checkpoints to reproduce paper results: [GeoMTL checkpoints on Hugging Face](https://huggingface.co/trust-tad/geomtl-vanilla).

Note: `geomtl` and the two classical baselines evaluate against `validation_data.txt`; `seg_ablation` evaluates against `test_data.txt` by default (`--test_ids_filename`), matching how each was originally reported.

## Installation

```bash
git clone https://github.com/TadieB/geoMTL.git
cd geoMTL
conda env create -f environment.yml
conda activate prithvi_py311
```

## Usage

All entry points use a subcommand for the model, followed by that model's arguments. Run `python scripts/train.py <model> --help` for the full argument list.

**Train**
```bash
python scripts/train.py geomtl --decoder_mode scratch \
    --data_path /path/to/data --model_weights /path/to/Prithvi_EO_V2_300M.pt

python scripts/train.py resnet_lstm --data_path /path/to/data

python scripts/train.py seg_ablation --finetune_full_model \
    --data_path /path/to/data --model_weights /path/to/Prithvi_EO_V2_300M.pt
```

**Evaluate** (quantitative metrics → JSON)
```bash
python scripts/evaluate.py geomtl --decoder_mode scratch \
    --checkpoint checkpoints/geomtl_scratch_best.pth \
    --model_weights /path/to/Prithvi_EO_V2_300M.pt --data_path /path/to/data \
    --output_file metrics_geomtl_scratch.json
```

**Inference** (qualitative panels + text reports)
```bash
python scripts/inference.py geomtl --decoder_mode scratch \
    --checkpoint checkpoints/geomtl_scratch_best.pth \
    --model_weights /path/to/Prithvi_EO_V2_300M.pt --data_path /path/to/data \
    --output_dir sample_outputs/geomtl_scratch
```

SLURM equivalents (see `slurm_jobs/*.sh` headers for more examples):
```bash
sbatch -J geomtl-scratch slurm_jobs/run_train.sh geomtl --decoder_mode scratch \
    --data_path /path/to/data --model_weights /path/to/Prithvi_EO_V2_300M.pt
```

## Reproducibility notes

- **`--itc_weight` (GeoMTL, pretrained decoder mode only):** this codebase defaults to `--itc_weight 1.0`. The paper's reported "GeoMTL (Pre-trained)" baseline numbers (Table I) were produced with the contrastive loss weighted at **0.5**. To reproduce those exact published numbers, pass `--itc_weight 0.5` explicitly; the `1.0` default here is intended as a fairer test of whether transfer learning helps, not a reproduction of the original run.
- **Qualitative inference (`--repetition_penalty`, `--seed`):** both default to `1.5` and `42` respectively, applied identically across every model, so generated captions and sampled chips are directly comparable across models. Override either flag if needed.
- **Decoder-mode confound:** the "pretrained" variant differs from "scratch" in three ways simultaneously (frozen vs. trainable decoder, decoder size, and presence of the auxiliary ITC text-encoder module), not decoder initialization alone. See the paper for discussion.

## Results

**Table I — Linguistic evaluation**

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|
| ResNet-50 + LSTM | 10.61 | 17.27 | 33.73 | 0.60 |
| 3D U-Net + LSTM | 10.21 | 17.48 | 33.88 | 0.47 |
| GeoMTL (pretrained decoder) | 3.07 | – | 27.04 | 4.12 |
| **GeoMTL (scratch, ours)** | **12.86** | **25.69** | **39.46** | 0.70 |

**Table II — Segmentation**

| Model | mIoU | Best Class IoU |
|---|---|---|
| Prithvi-EO-2 (IBM Terratorch) | 47.15 | 64.74 |
| Prithvi-EO-2 (reproduced, `seg_ablation`) | 47.94 | 65.08 |
| **GeoMTL (ours)** | 46.29 | **65.41** |

## Citation

```bibtex
@inproceedings{medimem2026geomtl,
  title     = {GEO-MTL: A Multi-task Framework for Joint Semantic Segmentation and Textual Scene Reporting in Remote Sensing},
  author    = {Medimem, Tadie B. and Melgani, Farid and Fiore, Sandro Luigi and Anantharaj, Valentine G.},
  booktitle = {IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  year      = {2026}
}
```

## License

[Add your chosen license here, e.g. MIT — currently no LICENSE file is present in this repo.]## Dataset

Due to size constraints, the dataset and checkpoints are hosted externally.

1. Download the [Multi-Temporal Crop Classification dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification/tree/main) and the corresponding [GeoMTL text annotations](https://huggingface.co/datasets/trust-tad/geomtl-dataset/tree/main).
2. Extract both into the same directory. It should contain `chips_df.csv`, `training_data.txt`, `validation_data.txt`, `test_data.txt`, `training_chips/`, `validation_chips/`, and the per-chip `.txt` / `.qual.txt` / `.clean.txt` caption files.
3. Pretrained checkpoints to reproduce paper results: [GeoMTL checkpoints on Hugging Face](https://huggingface.co/trust-tad/geomtl-vanilla).

Note: `geomtl` and the two classical baselines evaluate against `validation_data.txt`; `seg_ablation` evaluates against `test_data.txt` by default (`--test_ids_filename`), matching how each was originally reported.

## Installation

```bash
git clone https://github.com/TadieB/geoMTL.git
cd geoMTL
conda env create -f environment.yml
conda activate prithvi_py311
```

## Usage

All entry points use a subcommand for the model, followed by that model's arguments. Run `python scripts/train.py <model> --help` for the full argument list.

**Train**
```bash
python scripts/train.py geomtl --decoder_mode scratch \
    --data_path /path/to/data --model_weights /path/to/Prithvi_EO_V2_300M.pt

python scripts/train.py resnet_lstm --data_path /path/to/data

python scripts/train.py seg_ablation --finetune_full_model \
    --data_path /path/to/data --model_weights /path/to/Prithvi_EO_V2_300M.pt
```

**Evaluate** (quantitative metrics → JSON)
```bash
python scripts/evaluate.py geomtl --decoder_mode scratch \
    --checkpoint checkpoints/geomtl_scratch_best.pth \
    --model_weights /path/to/Prithvi_EO_V2_300M.pt --data_path /path/to/data \
    --output_file metrics_geomtl_scratch.json
```

**Inference** (qualitative panels + text reports)
```bash
python scripts/inference.py geomtl --decoder_mode scratch \
    --checkpoint checkpoints/geomtl_scratch_best.pth \
    --model_weights /path/to/Prithvi_EO_V2_300M.pt --data_path /path/to/data \
    --output_dir sample_outputs/geomtl_scratch
```

SLURM equivalents (see `slurm_jobs/*.sh` headers for more examples):
```bash
sbatch -J geomtl-scratch slurm_jobs/run_train.sh geomtl --decoder_mode scratch \
    --data_path /path/to/data --model_weights /path/to/Prithvi_EO_V2_300M.pt
```

## Reproducibility notes

- **`--itc_weight` (GeoMTL, pretrained decoder mode only):** this codebase defaults to `--itc_weight 1.0`. The paper's reported "GeoMTL (Pre-trained)" baseline numbers (Table I) were produced with the contrastive loss weighted at **0.5**. To reproduce those exact published numbers, pass `--itc_weight 0.5` explicitly; the `1.0` default here is intended as a fairer test of whether transfer learning helps, not a reproduction of the original run.
- **Qualitative inference (`--repetition_penalty`, `--seed`):** both default to `1.5` and `42` respectively, applied identically across every model, so generated captions and sampled chips are directly comparable across models. Override either flag if needed.
- **Decoder-mode confound:** the "pretrained" variant differs from "scratch" in three ways simultaneously (frozen vs. trainable decoder, decoder size, and presence of the auxiliary ITC text-encoder module), not decoder initialization alone. See the paper for discussion.

## Results

**Table I — Linguistic evaluation**

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|
| ResNet-50 + LSTM | 10.61 | 17.27 | 33.73 | 0.60 |
| 3D U-Net + LSTM | 10.21 | 17.48 | 33.88 | 0.47 |
| GeoMTL (pretrained decoder) | 3.07 | – | 27.04 | 4.12 |
| **GeoMTL (scratch, ours)** | **12.86** | **25.69** | **39.46** | 0.70 |

**Table II — Segmentation**

| Model | mIoU | Best Class IoU |
|---|---|---|
| Prithvi-EO-2 (IBM Terratorch) | 47.15 | 64.74 |
| Prithvi-EO-2 (reproduced, `seg_ablation`) | 47.94 | 65.08 |
| **GeoMTL (ours)** | 46.29 | **65.41** |

## Citation

```bibtex
@inproceedings{medimem2026geomtl,
  title     = {GEO-MTL: A Multi-task Framework for Joint Semantic Segmentation and Textual Scene Reporting in Remote Sensing},
  author    = {Medimem, Tadie B. and Melgani, Farid and Fiore, Sandro Luigi and Anantharaj, Valentine G.},
  booktitle = {IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  year      = {2026}
}
```

## License

[Add your chosen license here, e.g. MIT — currently no LICENSE file is present in this repo.]
