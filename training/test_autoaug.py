import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy 
from torch.utils.data import DataLoader
import os

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    best_learning_rate = 0.002
    best_batch_size = 16
    best_optimizer = 'SGD'
    best_dropout_rate = 0.3

    data_transforms = {
        'train': transforms.Compose([
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),  # Using AutoAugment
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = "processed_images"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=best_batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    def create_model(learning_rate, dropout_rate, optimizer_name, device):
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, len(class_names))
        )
        model_ft = model_ft.to(device)

        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(model_ft.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return model_ft, optimizer, criterion, scheduler

    model_ft, optimizer, criterion, scheduler = create_model(best_learning_rate, best_dropout_rate, best_optimizer, device)

    num_epochs = 25

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                scheduler.step()

    torch.save(model_ft.state_dict(), "model_with_autoaugment.pth")
    print(f'Final model training complete and saved.')

if __name__ == '__main__':
    main()


