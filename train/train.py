import sys
from pathlib import Path

# ==========================================================
# Add Project Root to Python Path
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
import tensorflow as tf

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

EPOCHS = 20

MODEL_PATH = "saved_models/attention_unet_fives.keras"

os.makedirs("saved_models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

(
    train_images,
    train_masks,
    val_images,
    val_masks,
    test_images,
    test_masks,
) = split_dataset()

train_dataset = create_dataset(
    train_images,
    train_masks,
    training=True,
)

val_dataset = create_dataset(
    val_images,
    val_masks,
)

# ==========================================================
# Build Model
# ==========================================================
from models.attention_unet import build_attention_unet

model = build_attention_unet()

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-4,
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=0.9,
    alpha=1e-6,
)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-5,
)

model.compile(
    optimizer=optimizer,

    loss=focal_tversky_loss,

    metrics=[
        dice_coefficient,
        iou_score,
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

# ==========================================================
# Callbacks
# ==========================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_dice_coefficient",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coefficient",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

]
# ==========================================================
# Train
# ==========================================================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ==========================================================
# Save Training History
# ==========================================================

np.save(
    "outputs/history.npy",
    history.history,
)

print("\nTraining History Saved!")

print("\nTraining Finished Successfully!")