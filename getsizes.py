import os
from PIL import Image


def get_image_sizes(folder_path):
    image_sizes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                width, height = img.size
                image_sizes.append((filename, width, height))
    return image_sizes


# Example usage
folder_path = "C:/Data/Python/IFCB/data/Diverse"  # Replace with the path to your folder
image_sizes = get_image_sizes(folder_path)

# Print the sizes of the images
for filename, width, height in image_sizes:
    print(f"Image: {filename}, Width: {width}, Height: {height}")
