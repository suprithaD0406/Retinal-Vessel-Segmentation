from pathlib import Path
from PIL import Image
import numpy as np

DATASET_PATH = Path("dataset")

print("=" * 60)
print("EMPTY MASK CHECK")
print("=" * 60)

for dataset in sorted(DATASET_PATH.iterdir()):
    if not dataset.is_dir():
        continue

    mask_folders = [f for f in dataset.rglob("*")
                    if f.is_dir() and f.name.lower() == "masks"]

    if not mask_folders:
        continue

    for folder in mask_folders:
        print(f"\n📂 {dataset.name}")

        empty = 0

        for mask_path in sorted(folder.iterdir()):
            mask = np.array(Image.open(mask_path).convert("L"))

            if np.count_nonzero(mask) == 0:
                empty += 1
                print(mask_path.name)

        print(f"Empty Masks : {empty}")