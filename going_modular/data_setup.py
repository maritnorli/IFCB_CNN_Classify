"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

NUM_WORKERS = os.cpu_count()


# Borrowed from Karin
def find_classes(dir):
    classes = os.listdir(dir)
    if ".DS_Store" in classes:
        classes.remove(".DS_Store")
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    simple_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    val_dir: Path to validation directory.
    test_dir: Path to testing directory.
          unclassifiable_dir: Path to unclassifiable images directory. Want to add a portion of images to new test and val datasets with unclassifiables
          but this applies mainly when I have determined thresholds I think
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, val_dataloader, test_dataloader, class_names, classes, class_to_idx). (missing unclass dataloader?)
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, val_dataloader, test_dataloader, class_names, classes, class_to_idx = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             val_dir = path/to/val_dir,
                             if added unclass_dir = path/to/unclass_dir
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
    # see if this matches with chatgpt
    # Use ImageFolder to create dataset(s) Maybe have no transforms here!
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=simple_transform)
    test_data = datasets.ImageFolder(test_dir, transform=simple_transform)
    # unclassifiable_dataset = datasets.ImageFolder(unclassifiable_dir, transform=simple_transform)

    # Get class names
    class_names = train_data.classes  # another classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    # unclass_dataloader = DataLoader(
    #     unclass_data,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=True,
    # )

    classes, class_to_idx = find_classes(train_dir)

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        class_names,
        classes,
        class_to_idx,
    )


#    return train_dataloader, validation_dataloader, test_dataloader, unclassifiable_dataloader, classes, class_to_idx
