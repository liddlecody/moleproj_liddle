import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [16, 32, 64]
optimizers = ['SGD', 'Adam']
dropout_rates = [0.3, 0.5]


results = {}

# Define the data transformations and datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "processed_images"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

def create_model(learning_rate, dropout_rate, optimizer_name, device):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, len(image_datasets['train'].classes))
    )

    model_ft = model_ft.to(device)

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model_ft.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    return model_ft, optimizer, criterion


for lr, batch_size, opt, dropout_rate in itertools.product(learning_rates, batch_sizes, optimizers, dropout_rates):
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

    model_ft, optimizer, criterion = create_model(lr, dropout_rate, opt, device)

    num_epochs = 10  

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1} with LR={lr}, Batch Size={batch_size}, Optimizer={opt}, Dropout={dropout_rate}")
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

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    results[(lr, batch_size, opt, dropout_rate)] = epoch_acc.item()


best_hyperparams = max(results, key=results.get)
print(f"Best Hyperparameters: Learning Rate={best_hyperparams[0]}, Batch Size={best_hyperparams[1]}, Optimizer={best_hyperparams[2]}, Dropout Rate={best_hyperparams[3]} with Accuracy={results[best_hyperparams]:.4f}")
