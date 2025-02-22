#script to evaluate the captioning model on a set of images, allowing user to specify model path and image directory.

import torch
from torchvision import transforms
from src.captioning.model import CNNtoRNN
from src.captioning.utils import load_image
from PIL import Image


def evaluate_image(image_path, model, vocabulary, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        features = model.encoderCNN(image)
        caption = model.caption_image(image, vocabulary)

    return " ".join(caption)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your trained model and vocabulary here
    embed_size = 256
    hidden_size = 256
    vocab_size = 1000  # Replace with your actual vocabulary size
    model = CNNtoRNN(embed_size, hidden_size, vocab_size).to(device)
    model.load_state_dict(torch.load("path_to_your_model.pth"))

    # Load your vocabulary here
    # vocabulary = ...

    image_path = input("Enter the path to the image: ")
    caption = evaluate_image(image_path, model, vocabulary, device)
    print(f"Generated caption: {caption}")
