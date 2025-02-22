import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.captions = self.load_captions(captions_file)
        self.img_paths = list(self.captions.keys())

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_paths[idx])
        image = Image.open(img_path).convert('RGB')
        caption = self.captions[self.img_paths[idx]]

        if self.transform:
            image = self.transform(image)

        return image, caption

    def load_captions(self, captions_file):
        captions = {}
        with open(captions_file, 'r') as f:
            for line in f:
                img_path, caption = line.strip().split('\t')
                captions[img_path] = caption
        return captions


def get_loader(root_folder, annotation_file, transform, batch_size=32, num_workers=4, shuffle=True, pin_memory=True):
    dataset = ImageCaptionDataset(root_folder, annotation_file, transform=transform)

    pad_idx = 0  # Assuming 0 is the pad token

    def collate_fn(batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)
        return imgs, targets

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    return loader
