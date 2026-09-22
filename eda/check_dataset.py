from pathlib import Path
from PIL import Image

DATASET_PATH = Path("dataset")

total_images = 0
corrupted = []

for dataset in DATASET_PATH.iterdir():
    if not dataset.is_dir():
        continue

    for folder in dataset.rglob("*"):
        if folder.is_dir() and folder.name.lower() in ["images", "masks", "roi_masks"]:
            for file in folder.iterdir():
                try:
                    Image.open(file).verify()
                    total_images += 1
                except Exception:
                    corrupted.append(str(file))

print("=" * 50)
print("DATASET REPORT")
print("=" * 50)
print(f"Total Files Checked : {total_images}")
print(f"Corrupted Files     : {len(corrupted)}")

if corrupted:
    print("\nCorrupted Files:")
    for file in corrupted:
        print(file)
else:
    print("\n✅ No corrupted files found.")