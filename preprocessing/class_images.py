import os
import shutil


data_dir = "processed_images"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")


class_names = ['benign', 'malignant']

for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

#move images to the appropriate class folder
def move_images_to_class_folders(source_dir, class_names):
    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg"): 
            for class_name in class_names:
                if class_name in filename:
                    dest_dir = os.path.join(source_dir, class_name)
                    shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))
                    print(f"Moved {filename} to {dest_dir}")
                    break


move_images_to_class_folders(train_dir, class_names)
move_images_to_class_folders(val_dir, class_names)

print("Images have been organized into their respective class folders.")