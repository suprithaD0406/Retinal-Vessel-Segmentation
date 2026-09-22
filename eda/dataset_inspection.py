from pathlib import Path

DATASET_PATH = Path("dataset")

print("=" * 50)
print("RETINAL DATASET INSPECTION")
print("=" * 50)

for dataset in DATASET_PATH.iterdir():
    if dataset.is_dir():
        print(f"\n📂 {dataset.name}")

        for folder in dataset.iterdir():
            if folder.is_dir():
                count = len(list(folder.glob("*")))
                print(f"   ├── {folder.name} : {count} files")