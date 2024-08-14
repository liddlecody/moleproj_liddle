from PIL import Image
import os
from torchvision import transforms

input_dir = "isic_images"
output_dir = "processed_images"

os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),    # Resize images to 224x224
    transforms.ToTensor(),            # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
                         std=[0.229, 0.224, 0.225])
])

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        try:
            img = Image.open(os.path.join(input_dir, filename))

            transformed_img = transform(img)
            
            transformed_img = transforms.ToPILImage()(transformed_img)

            transformed_img.save(os.path.join(output_dir, filename))


            print(f"Processed and saved {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")