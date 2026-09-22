from pathlib import Path

# ==========================================================
# Dataset Path
# ==========================================================

DATASET_PATH = Path("dataset")

# ==========================================================
# Choose Dataset
# ==========================================================

# Original Dataset
# DATASET_NAME = "FIVES"

# CLAHE Enhanced Dataset
DATASET_NAME = "FIVES_CLAHE"

# ==========================================================
# Load Dataset
# ==========================================================

def load_fives_paths():

    image_dir = DATASET_PATH / DATASET_NAME / "images"
    mask_dir = DATASET_PATH / DATASET_NAME / "masks"

    invalid_masks = {
        "train_447_G.png",
        "train_448_G.png",
    }

    image_paths = []
    mask_paths = []

    for image_path in sorted(image_dir.glob("*")):

        mask_path = mask_dir / image_path.name

        if mask_path.name in invalid_masks:
            continue

        if not mask_path.exists():
            continue

        image_paths.append(str(image_path))
        mask_paths.append(str(mask_path))

    return image_paths, mask_paths


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    images, masks = load_fives_paths()

    print("=" * 50)
    print("DATASET")
    print("=" * 50)

    print("Dataset Used :", DATASET_NAME)
    print("Total Images :", len(images))
    print("Total Masks  :", len(masks))

    print("\nFirst Image")
    print(images[0])

    print("\nFirst Mask")
    print(masks[0])