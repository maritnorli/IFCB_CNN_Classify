import os
import random
import shutil
import torch
import numpy as np
from pathlib import Path


def split_images(
    folder_path,
    train_dir,
    val_dir,
    test_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
):
    # Create Train, Val, and Test folders
    train_folder = os.path.join(train_dir, os.path.basename(folder_path))
    val_folder = os.path.join(val_dir, os.path.basename(folder_path))
    test_folder = os.path.join(test_dir, os.path.basename(folder_path))

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    num_images = len(image_files)

    # Calculate number of images for Train, Val, and Test
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    num_test = num_images - num_train - num_val

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Copy images to Train, Val, and Test folders
    for i, image_file in enumerate(image_files):
        src = os.path.join(folder_path, image_file)
        if i < num_train:
            dst = os.path.join(train_folder, image_file)
        elif i < num_train + num_val:
            dst = os.path.join(val_folder, image_file)
        else:
            dst = os.path.join(test_folder, image_file)
        shutil.copy(src, dst)

    print(
        "Splitting completed. Train folder contains {} images, Val folder contains {} images, Test folder contains {} images.".format(
            num_train, num_val, num_test
        )
    )


def process_folders(images_dir, train_dir, val_dir, test_dir):
    for folder_name in os.listdir(images_dir):
        folder_path = os.path.join(images_dir, folder_name)
        if os.path.isdir(folder_path):
            split_images(folder_path, train_dir, val_dir, test_dir)


# Example usage
# process_folders("/path/to/images", "/path/to/train", "/path/to/val", "/path/to/test")


## To make smaller datasets:


import os
import random
import shutil


def split_images_small(
    folder_path,
    train_dir,
    val_dir,
    test_dir,
    max_num_images=10,
    split_ratio=(0.8, 0.1, 0.1),
):
    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    total_images = len(image_files)

    # Check if the folder contains fewer than 5 images
    if total_images < 7:
        print(f"Folder '{folder_path}' contains fewer than 7 images. Skipping.")
        return

    # Select a subset of images randomly, with a maximum of `max_num_images`
    random.shuffle(image_files)
    num_images_to_use = min(total_images, max_num_images)
    selected_images = image_files[:num_images_to_use]

    # Calculate the number of images for Train, Validation, and Test based on split ratio
    num_train = int(num_images_to_use * split_ratio[0])
    num_val = int(num_images_to_use * split_ratio[1])
    num_test = num_images_to_use - num_train - num_val

    # Create Train, Validation, and Test folders
    train_folder = os.path.join(train_dir, os.path.basename(folder_path))
    val_folder = os.path.join(val_dir, os.path.basename(folder_path))
    test_folder = os.path.join(test_dir, os.path.basename(folder_path))

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Copy images to Train, Validation, and Test folders
    for i, image_file in enumerate(selected_images):
        src = os.path.join(folder_path, image_file)
        if i < num_train:
            dst = os.path.join(train_folder, image_file)
        elif i < num_train + num_val:
            dst = os.path.join(val_folder, image_file)
        else:
            dst = os.path.join(test_folder, image_file)
        shutil.copy(src, dst)

    print(
        f"Splitting completed. Train folder contains {num_train} images, "
        f"Validation folder contains {num_val} images, "
        f"Test folder contains {num_test} images."
    )


def process_folders_small(
    images_dir,
    train_dir,
    val_dir,
    test_dir,
    max_num_images=10,
    split_ratio=(0.8, 0.1, 0.1),
):
    for folder_name in os.listdir(images_dir):
        folder_path = os.path.join(images_dir, folder_name)
        if os.path.isdir(folder_path):
            split_images_small(
                folder_path, train_dir, val_dir, test_dir, max_num_images, split_ratio
            )


# Example usage

# process_folders_small(
#     "/path/to/images_dir",
#     "/path/to/train_dir",
#     "/path/to/val_dir",
#     "/path/to/test_dir",
#     max_num_images=10,
#     split_ratio=(0.8, 0.1, 0.1)
# )
