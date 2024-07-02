import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # Import matplotlib for inline image display
from torchvision import transforms
from torchvision import datasets


class CustomResizeAndReflectPad:
    def __call__(self, img):
        return resize_and_reflect_pad(img)


def resize_and_reflect_pad(img):
    desired_size = 244
    width, height = img.size
    aspect_ratio = width / float(height)

    if width > height:
        new_height = int(desired_size / aspect_ratio)
        resized_img = img.resize((desired_size, new_height), Image.LANCZOS)
    else:
        new_width = int(desired_size * aspect_ratio)
        resized_img = img.resize((new_width, desired_size), Image.LANCZOS)

    # Calculate padding to fill up to 244x244
    pad_width = (desired_size - resized_img.size[0]) // 2
    pad_height = (desired_size - resized_img.size[1]) // 2

    # Handle the case where the size after padding might not be even
    padding = (
        pad_width,
        pad_height,
        desired_size - resized_img.size[0] - pad_width,
        desired_size - resized_img.size[1] - pad_height,
    )

    # Apply reflection padding using torchvision's pad function
    padded_img = TF.pad(resized_img, padding, padding_mode="reflect")

    # Ensure the final size is exactly 244x244
    padded_img = TF.center_crop(padded_img, (desired_size, desired_size))

    # Print the size for debugging
    print(f"Final padded image size: {padded_img.size}")

    # Mirror the image (if necessary, otherwise remove this step)
    mirrored_img = TF.vflip(padded_img)

    return mirrored_img
