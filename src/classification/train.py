import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from src.classification.model import CNNModel
import os
import argparse

def train_model(data_dir, save_dir, arch='alexnet', gpu=False, learning_rate=0.001, hidden_units=512, epochs=10):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    model = CNNModel(arch=arch, hidden_units=hidden_units, num_classes=len(class_names))
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)
    else:
        device = torch.device("cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
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

    torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classifier Training')
    parser.add_argument('data_dir', help='Directory containing training and validation images')
    parser.add_argument('--save_dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', default='alexnet', choices=['vgg19', 'resnet34', 'alexnet'], help='CNN model to use')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier layer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_model(args.data_dir, args.save_dir, args.arch, args.gpu, args.learning_rate, args.hidden_units, args.epochs)
