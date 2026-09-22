from pathlib import Path
from PIL import Image
import numpy as np

DATASET_PATH = Path("dataset")

print("=" * 70)
print("IMAGE & MASK PROPERTIES")
print("=" * 70)

for dataset in DATASET_PATH.iterdir():
    if not dataset.is_dir():
        continue

    print(f"\n📂 {dataset.name}")

    image_folder = None
    mask_folder = None

    for folder in dataset.rglob("*"):
        name = folder.name.lower()
        if folder.is_dir():
            if name == "images":
                image_folder = folder
            elif name == "masks":
                mask_folder = folder

    if image_folder:
        img = Image.open(next(image_folder.iterdir()))
        arr = np.array(img)

        print(f"Image Format : {img.format}")
        print(f"Image Mode   : {img.mode}")
        print(f"Image Shape  : {arr.shape}")

    if mask_folder:
        mask = Image.open(next(mask_folder.iterdir()))
        arr = np.array(mask)

        print(f"Mask Format  : {mask.format}")
        print(f"Mask Mode    : {mask.mode}")
        print(f"Mask Shape   : {arr.shape}")
        print(f"Mask Values  : {np.unique(arr)[:10]}")