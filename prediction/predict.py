import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ==========================================================
# Add Project Root to Python Path
# ==========================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# ==========================================================
# Import Custom Losses
# ==========================================================

from utils.losses import (
    bce_dice_loss,
    dice_coefficient,
    iou_score
)

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "saved_models/unet_fives.keras"

IMAGE_SIZE = (256, 256)

# ==========================================================
# Load Trained Model
# ==========================================================

print("=" * 60)
print("Loading Model...")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score": iou_score
    }
)

print("Model Loaded Successfully!")

# ==========================================================
# Image Path
# ==========================================================

image_path = input("\nEnter image path: ")

# ==========================================================
# Read Image
# ==========================================================

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found!")

original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(original, IMAGE_SIZE)

image = image.astype(np.float32) / 255.0

input_image = np.expand_dims(image, axis=0)

# ==========================================================
# Prediction
# ==========================================================

prediction = model.predict(input_image, verbose=0)[0]

prediction = (prediction > 0.5).astype(np.uint8)

# ==========================================================
# Display Result
# ==========================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(prediction.squeeze(), cmap="gray")
plt.title("Predicted Vessel Mask")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nPrediction Completed Successfully!")