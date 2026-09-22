import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.models import load_model

# ==========================================================
# Add Project Root to Python Path
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

# ==========================================================
# Import Project Files
# ==========================================================

from preprocessing.split_dataset import split_dataset
from training.data_pipeline import create_dataset

from utils.losses import (
    focal_tversky_loss,
    dice_coefficient,
    iou_score,
)

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "saved_models/attention_unet_fives.keras"

# ==========================================================
# Load Test Dataset
# ==========================================================

(
    train_images,
    train_masks,
    val_images,
    val_masks,
    test_images,
    test_masks,
) = split_dataset()

test_dataset = create_dataset(
    test_images,
    test_masks,
    training=False,
)

# ==========================================================
# Load Model
# ==========================================================

print("=" * 60)
print("Loading Trained Model...")
print("=" * 60)

model = load_model(
    MODEL_PATH,
    custom_objects={
        "focal_tversky_loss": focal_tversky_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score": iou_score,
    },
)

print("Model Loaded Successfully!\n")

# ==========================================================
# Evaluate
# ==========================================================

print("=" * 60)
print("Evaluating Model...")
print("=" * 60)

results = model.evaluate(
    test_dataset,
    verbose=1,
)

print("\nTest Results")
print("-" * 40)

for name, value in zip(model.metrics_names, results):
    print(f"{name:<20}: {value:.4f}")

# ==========================================================
# Pixel-wise Precision & Recall
# ==========================================================

y_true = []
y_pred = []

for images, masks in test_dataset:

    predictions = model.predict(
        images,
        verbose=0,
    )

    predictions = (
        predictions > 0.5
    ).astype(np.uint8)

    y_true.extend(
        masks.numpy().astype(np.uint8).flatten()
    )

    y_pred.extend(
        predictions.flatten()
    )

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

print("\nAdditional Metrics")
print("-" * 40)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

print("\nEvaluation Completed Successfully!")