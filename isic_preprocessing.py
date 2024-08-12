from PIL import Image
import os
from torchvision import transforms

# Define the directory where images are stored
input_dir = "isic_images"
output_dir = "processed_images"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),    # Resize images to 224x224
    transforms.ToTensor(),            # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
                         std=[0.229, 0.224, 0.225])
])

# Optionally, you can add data augmentation
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Loop through all images in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        try:
            # Load the image
            img = Image.open(os.path.join(input_dir, filename))

            # Apply the transformation
            transformed_img = transform(img)
            
            # Convert back to PIL Image for saving (optional if not using PyTorch)
            transformed_img = transforms.ToPILImage()(transformed_img)

            # Save the processed image
            transformed_img.save(os.path.join(output_dir, filename))

            # Optionally, apply augmentation and save augmented images
            '''aug_img = augmentation(img)
            aug_img = transforms.ToPILImage()(aug_img)
            aug_filename = f"aug_{filename}"
            aug_img.save(os.path.join(output_dir, aug_filename))'''

            print(f"Processed and saved {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")