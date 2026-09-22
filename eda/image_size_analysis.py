from pathlib import Path
from PIL import Image
from collections import Counter

DATASET_PATH = Path("dataset")

print("=" * 60)
print("IMAGE SIZE ANALYSIS")
print("=" * 60)

for dataset in DATASET_PATH.iterdir():
    if not dataset.is_dir():
        continue

    image_folder = None

    for folder in dataset.rglob("*"):
        if folder.is_dir() and folder.name.lower() == "images":
            image_folder = folder
            break

    if image_folder is None:
        continue

    sizes = []

    for img_path in image_folder.iterdir():
        with Image.open(img_path) as img:
            sizes.append(img.size)

    print(f"\n📂 {dataset.name}")
    print(f"Total Images : {len(sizes)}")

    size_count = Counter(sizes)

    for size, count in size_count.items():
        print(f"{size} --> {count}")