from pathlib import Path

DATASET_PATH = Path("dataset")

print("=" * 60)
print("IMAGE - MASK PAIR CHECK")
print("=" * 60)

for dataset in sorted(DATASET_PATH.iterdir()):
    if not dataset.is_dir():
        continue

    image_folder = None
    mask_folder = None

    for folder in dataset.rglob("*"):
        if folder.is_dir():
            if folder.name.lower() == "images":
                image_folder = folder
            elif folder.name.lower() == "masks":
                mask_folder = folder

    if image_folder is None or mask_folder is None:
        continue

    print(f"\n📂 {dataset.name}")

    print(f"Images : {len(list(image_folder.iterdir()))}")
    print(f"Masks  : {len(list(mask_folder.iterdir()))}")

    if len(list(image_folder.iterdir())) == len(list(mask_folder.iterdir())):
        print("✅ Counts Match")
    else:
        print("❌ Counts Do Not Match")