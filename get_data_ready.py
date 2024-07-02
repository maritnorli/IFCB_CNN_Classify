from pathlib import Path
from data_ready_functions import process_folders_small


# Small/med/test classifier dataset paths:
# data_path = Path("C:\Data\Python\IFCB\data")
# image_path = Path("C:\Data\Python\IFCB\data\IFCB_test_train")

# # # Setup Dirs
# # train_dir = image_path / "Train_small"
# # val_dir = image_path / "Val_small"
# # test_dir = image_path / "Test_small"

# # Setup Dirs
# train_dir = image_path / "Train_med"
# val_dir = image_path / "Val_med"
# test_dir = image_path / "Test_med"

# images_dir = Path("C:\Data\Python\Classifier_rev")

# process_folders_small(
#     images_dir,
#     train_dir,
#     val_dir,
#     test_dir,
#     max_num_images=50,
#     split_ratio=(0.7, 0.15, 0.15),
# )

# Small/med/test classifier dataset paths:
data_path = Path("C:\Data\Python\IFCB\data")
image_path = Path("C:\Data\Python\IFCB\data\IFCB_test_train")

# # Setup Dirs
# train_dir = image_path / "Train_small"
# val_dir = image_path / "Val_small"
# test_dir = image_path / "Test_small"

# Setup Dirs
train_dir = image_path / "Train"
val_dir = image_path / "Val"
test_dir = image_path / "Test"

images_dir = Path("C:\Data\Python\Classifier_rev")

process_folders_small(
    images_dir,
    train_dir,
    val_dir,
    test_dir,
    max_num_images=5000,
    split_ratio=(0.7, 0.15, 0.15),
)
