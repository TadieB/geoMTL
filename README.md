# GeoMTL: Multi-Task Learning for Earth Observation

This repository contains the official PyTorch implementation for our IEEE IGARSS presentation on **GeoMTL**. GeoMTL is a multi-task learning architecture designed to perform simultaneous Land Cover Segmentation and Satellite Image Captioning (Scene reporting) using multi-temporal Satellite Data.

## 🗂️ Repository Structure

*   `models/`: Contains all PyTorch model definitions (GeoMTL, Baselines, Prithvi Encoder).
*   `scripts/`: Unified execution scripts for training, evaluation, and inference.
*   `slurm_jobs/`: Bash scripts for submitting jobs to a SLURM cluster.
*   `sample_outputs/`: Examples of generated NASA-style visualization panels and text reports.

## 💾 Dataset

Due to size constraints, the dataset is hosted externally. 
1. Download the Multi-Temporal Crop Dataset from [Insert Link Here].
2. Extract the dataset into your preferred directory.
3. The dataset should contain `chips_df.csv`, `test_data.txt`, `training_data.txt`, `validation_data.txt`,`training_chips/`, `validation_chips/`, and the corresponding 
 and  `.txt` caption files.

## ⚙️ Installation

We provide a Conda environment file to ensure full reproducibility. 

```bash
git clone [https://github.com/yourusername/GeoMTL-IGARSS.git](https://github.com/yourusername/GeoMTL-IGARSS.git)
cd GeoMTL-IGARSS
conda env create -f environment.yml
conda activate prithvi_py311
