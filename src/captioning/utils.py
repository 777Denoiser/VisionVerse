#utility processing for data preprocessing, loading pre-trained CNN models and generating captions

import torch
from torchvision import transforms
from PIL import Image

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    image = transform(image).unsqueeze(0)
    return image

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_vocabulary(vocab_file):
    with open(vocab_file, 'r') as f:
        return json.load(f)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
