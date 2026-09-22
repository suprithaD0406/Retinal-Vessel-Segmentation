from sklearn.model_selection import train_test_split
from preprocessing.dataset_loader import load_fives_paths


def split_dataset():
    image_paths, mask_paths = load_fives_paths()

    # 80% Train, 20% Temp
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    # Split remaining 20% into Validation and Test
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images,
        temp_masks,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    return (
        train_images,
        train_masks,
        val_images,
        val_masks,
        test_images,
        test_masks,
    )


if __name__ == "__main__":

    (
        train_images,
        train_masks,
        val_images,
        val_masks,
        test_images,
        test_masks,
    ) = split_dataset()

    print("=" * 50)
    print("DATASET SPLIT")
    print("=" * 50)

    print("Training Images   :", len(train_images))
    print("Validation Images :", len(val_images))
    print("Testing Images    :", len(test_images))

    print("\nTotal :", len(train_images) + len(val_images) + len(test_images))