from pathlib import Path
from PIL import Image
import numpy as np

DATASET_PATH = Path("dataset")

print("=" * 60)
print("VESSEL COVERAGE ANALYSIS")
print("=" * 60)

for dataset in DATASET_PATH.iterdir():
    if not dataset.is_dir():
        continue

    mask_folder = None

    for folder in dataset.rglob("*"):
        if folder.is_dir() and folder.name.lower() == "masks":
            mask_folder = folder
            break

    if mask_folder is None:
        continue

    percentages = []

    for mask_path in mask_folder.iterdir():
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)

        vessel_pixels = np.sum(mask > 127)
        total_pixels = mask.size

        percentages.append((vessel_pixels / total_pixels) * 100)

    print(f"\n📂 {dataset.name}")
    print(f"Average Vessel Coverage : {np.mean(percentages):.2f}%")
    print(f"Minimum                : {np.min(percentages):.2f}%")
    print(f"Maximum                : {np.max(percentages):.2f}%")