#script for performing image classification prediction on new images

import torch
from torchvision import transforms
from PIL import Image
import json
import argparse
from src.classification.model import CNNModel

def load_checkpoint(checkpoint_path, arch='alexnet', hidden_units=512, num_classes=102):
    model = CNNModel(arch=arch, hidden_units=hidden_units, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def process_image(image_path):
    pil_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(pil_image)

def predict(image_path, checkpoint_path, top_k=5, category_names=None, gpu=False, arch='alexnet', hidden_units=512):
    model = load_checkpoint(checkpoint_path, arch=arch, hidden_units=hidden_units)
    image = process_image(image_path)
    image = image.unsqueeze(0)

    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)
        image = image.to(device)
    else:
        device = torch.device("cpu")

    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probs, top_classes = torch.topk(probabilities, top_k)

    top_probs = top_probs.cpu().numpy().tolist()[0]
    top_classes = top_classes.cpu().numpy().tolist()[0]

    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_names = [cat_to_name[str(index + 1)] for index in top_classes]
    else:
        top_names = top_classes

    return top_probs, top_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classifier - Prediction')
    parser.add_argument('input_image_path', help='Path to input image to be classified')
    parser.add_argument('checkpoint', help='Path to model checkpoint file (.pth)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to return')
    parser.add_argument('--category_names', default='cat_to_name.json', help='Category numbers to names file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--arch', default='alexnet', help='Model Architecture')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden Units')
    args = parser.parse_args()

    top_probs, top_names = predict(args.input_image_path, args.checkpoint, top_k=args.top_k,
                                   category_names=args.category_names, gpu=args.gpu, arch=args.arch, hidden_units=args.hidden_units)

    for i in range(len(top_probs)):
        print(f"{top_names[i]}: {top_probs[i]:.4f}")
