import os
import shutil

# Define paths
data_dir = "processed_images"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# Create class subdirectories
class_names = ['benign', 'malignant']

for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

# Function to move images to the appropriate class folder
def move_images_to_class_folders(source_dir, class_names):
    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg"):  # Adjust this if your images have different extensions
            for class_name in class_names:
                if class_name in filename:
                    # Move the file to the corresponding class folder
                    dest_dir = os.path.join(source_dir, class_name)
                    shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))
                    print(f"Moved {filename} to {dest_dir}")
                    break

# Move images to train/class_name folders
move_images_to_class_folders(train_dir, class_names)

# Move images to val/class_name folders
move_images_to_class_folders(val_dir, class_names)

print("Images have been organized into their respective class folders.")