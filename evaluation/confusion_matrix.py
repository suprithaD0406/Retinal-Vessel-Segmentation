import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

# ==========================================================
# Add Project Root
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

# ==========================================================
# Imports
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

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# Load Model
# ==========================================================

model = load_model(
    MODEL_PATH,
    custom_objects={
        "focal_tversky_loss": focal_tversky_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score": iou_score,
    },
)

# ==========================================================
# Load Test Dataset
# ==========================================================

(
    _,
    _,
    _,
    _,
    test_images,
    test_masks,
) = split_dataset()

test_dataset = create_dataset(
    test_images,
    test_masks,
    training=False,
)

# ==========================================================
# Predictions
# ==========================================================

y_true = []
y_pred = []

for images, masks in test_dataset:

    preds = model.predict(images, verbose=0)
    preds = (preds > 0.5).astype(np.uint8)

    y_true.extend(masks.numpy().flatten())
    y_pred.extend(preds.flatten())

y_true = np.array(y_true, dtype=np.uint8)
y_pred = np.array(y_pred, dtype=np.uint8)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_true, y_pred)

# ==========================================================
# Plot
# ==========================================================

fig, ax = plt.subplots(figsize=(5, 5))

im = ax.imshow(cm, cmap="Blues")

classes = ["Background", "Vessel"]

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(classes, fontsize=10)
ax.set_yticklabels(classes, fontsize=10)

ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title("Confusion Matrix", fontsize=13)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            f"{cm[i,j]:,}",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)

cbar.ax.tick_params(labelsize=9)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("=" * 50)
print("Confusion Matrix Saved Successfully")
print("=" * 50)
print(cm)