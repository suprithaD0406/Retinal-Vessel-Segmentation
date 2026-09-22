import cv2
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

INPUT_DATASET = Path("dataset/FIVES")
OUTPUT_DATASET = Path("dataset/FIVES_CLAHE")

IMAGE_INPUT = INPUT_DATASET / "images"
MASK_INPUT = INPUT_DATASET / "masks"

IMAGE_OUTPUT = OUTPUT_DATASET / "images"
MASK_OUTPUT = OUTPUT_DATASET / "masks"

IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
MASK_OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Check Dataset
# ==========================================================

print("=" * 50)
print("INPUT IMAGE FOLDER :", IMAGE_INPUT)
print("INPUT MASK FOLDER  :", MASK_INPUT)
print("=" * 50)

if not IMAGE_INPUT.exists():
    print("ERROR: Image folder not found!")
    exit()

if not MASK_INPUT.exists():
    print("ERROR: Mask folder not found!")
    exit()

# ==========================================================
# CLAHE
# ==========================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)

# ==========================================================
# Process Image
# ==========================================================

def process_image(src, dst):

    image = cv2.imread(str(src))

    if image is None:
        print("Failed :", src.name)
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green = image[:, :, 1]

    green = clahe.apply(green)

    image[:, :, 1] = green

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(dst), image)

# ==========================================================
# Copy Mask
# ==========================================================

def copy_mask(src, dst):

    mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Failed :", src.name)
        return

    cv2.imwrite(str(dst), mask)

# ==========================================================
# Process Images
# ==========================================================

image_files = sorted(IMAGE_INPUT.glob("*"))
mask_files = sorted(MASK_INPUT.glob("*"))

print(f"Found {len(image_files)} images")
print(f"Found {len(mask_files)} masks")

print("\nStarting image processing...\n")

image_count = 0

for image_path in image_files:

    print(f"[{image_count+1}/{len(image_files)}] {image_path.name}")

    process_image(
        image_path,
        IMAGE_OUTPUT / image_path.name,
    )

    image_count += 1

print("\nImages Finished.\n")

mask_count = 0

for mask_path in mask_files:

    copy_mask(
        mask_path,
        MASK_OUTPUT / mask_path.name,
    )

    mask_count += 1

print("\nMasks Finished.\n")

print("=" * 50)
print("CLAHE PREPROCESSING COMPLETED")
print("=" * 50)
print("Images :", image_count)
print("Masks  :", mask_count)
print("Saved To :", OUTPUT_DATASET)