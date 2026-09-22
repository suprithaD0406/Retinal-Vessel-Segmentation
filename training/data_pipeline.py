import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from preprocessing.split_dataset import split_dataset

# ==========================================================
# Configuration
# ==========================================================

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8

# ==========================================================
# Load Image & Mask
# ==========================================================

def load_image(image_path, mask_path):

    # Read Image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    # Read Mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(
        mask,
        IMAGE_SIZE,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    mask = tf.cast(mask > 127, tf.float32)

    return image, mask


# ==========================================================
# Data Augmentation
# ==========================================================

def augment(image, mask):

    # Random Horizontal Flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Random Vertical Flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Random Rotation (0°, 90°, 180°, 270°)
    k = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )

    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    # Random Brightness
    image = tf.image.random_brightness(
        image,
        max_delta=0.10,
    )

    # Random Contrast
    image = tf.image.random_contrast(
        image,
        lower=0.90,
        upper=1.10,
    )

    # Keep pixel values in range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


# ==========================================================
# Create Dataset
# ==========================================================

def create_dataset(image_paths, mask_paths, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, mask_paths)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:

        dataset = dataset.shuffle(
            len(image_paths),
            reshuffle_each_iteration=True,
        )

        dataset = dataset.map(
            augment,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE,
    )

    return dataset


# ==========================================================
# Test Pipeline
# ==========================================================

if __name__ == "__main__":

    (
        train_images,
        train_masks,
        val_images,
        val_masks,
        test_images,
        test_masks,
    ) = split_dataset()

    train_dataset = create_dataset(
        train_images,
        train_masks,
        training=True,
    )

    print("=" * 50)
    print("TensorFlow Dataset Created")
    print("=" * 50)

    print("Training Samples   :", len(train_images))
    print("Validation Samples :", len(val_images))
    print("Testing Samples    :", len(test_images))

    for images, masks in train_dataset.take(1):

        print("\nBatch Image Shape :", images.shape)
        print("Batch Mask Shape  :", masks.shape)

        print(
            "\nImage Range :",
            float(tf.reduce_min(images)),
            "to",
            float(tf.reduce_max(images)),
        )

        print(
            "Mask Range  :",
            float(tf.reduce_min(masks)),
            "to",
            float(tf.reduce_max(masks)),
        )