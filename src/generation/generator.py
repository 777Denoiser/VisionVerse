#contains core logic for text to image generation using CLIP and SIREN based on deep daze.

import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
import clip
from src.generation.utils import SIREN, get_init_images


class ImageGenerator(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.siren = SIREN(
            dim_in=2,
            dim_hidden=256,
            dim_out=3,
            num_layers=5,
            w0_initial=30.
        )

    def forward(self, text, num_iterations=500, lr=1e-2):
        text_inputs = clip.tokenize([text]).to(next(self.parameters()).device)
        target_features = self.clip_model.encode_text(text_inputs)

        init_images = get_init_images((224, 224))
        images = init_images.clone().requires_grad_()

        optimizer = torch.optim.Adam([images], lr=lr)

        for i in range(num_iterations):
            optimizer.zero_grad()

            image_features = self.clip_model.encode_image(images)
            loss = -torch.cosine_similarity(image_features, target_features).mean()

            if i % 50 == 0:
                print(f'Iteration {i}, Loss: {loss.item()}')

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                images.clamp_(0., 1.)

        return images


def generate_image(text, iterations=500, lr=1e-2, device='cuda'):
    clip_model, _ = clip.load("ViT-B/32", device=device)
    generator = ImageGenerator(clip_model).to(device)

    with torch.no_grad():
        generated_images = generator(text, num_iterations=iterations, lr=lr)

    # Convert to PIL Image
    transform = T.ToPILImage()
    image = transform(generated_images[0].cpu())

    return image

