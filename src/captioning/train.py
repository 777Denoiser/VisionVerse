#script for training and fine-tuning the classification model using a custom dataset of my own

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from src.captioning.model import CNNtoRNN
from src.captioning.utils import save_checkpoint, load_checkpoint

def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

def main():
    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = 1000  # Replace with your actual vocabulary size
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 10

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss, and optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the pad token
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load data (you need to implement your custom dataset)
    # train_dataset = YourCustomDataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Train the model
    train(model, train_loader, criterion, optimizer, device, num_epochs)

    # Save the model checkpoint
    save_checkpoint(model, optimizer, 'caption_model.pth')

if __name__ == '__main__':
    main()
