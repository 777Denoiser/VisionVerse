import json
import torch
from torchvision import transforms
from PIL import Image

def load_category_names(category_file):
    with open(category_file, 'r') as f:
        return json.load(f)

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def save_checkpoint(model, optimizer, class_to_idx, filepath):
    checkpoint = {
        'arch': model.__class__.__name__,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    return checkpoint
