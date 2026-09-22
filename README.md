# Retinal Vessel Segmentation

Deep learning-based retinal blood vessel segmentation using **U-Net** and **Attention U-Net**.

## Features

- Retinal image preprocessing
- U-Net-based vessel segmentation
- Attention U-Net-based vessel segmentation
- Model evaluation
- Prediction on new retinal images
- Dataset analysis and visualization

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- U-Net
- Attention U-Net

## Dataset

Retinal Vessel Segmentation Datasets Collection.

The dataset contains retinal fundus images and corresponding vessel segmentation masks from multiple datasets, including:

- FIVES
- DRIVE
- STARE
- CHASE
- HRF

The dataset is not included in this repository because of its size.

## Project Structure

```text
Retinal-Vessel-Segmentation/
│
├── eda/
├── evaluation/
├── prediction/
├── preprocessing/
├── train/
├── training/
├── utils/
│
├── main.py
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md