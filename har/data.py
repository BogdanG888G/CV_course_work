from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T


def image_transforms(size=224, train=False):
    operations = (
        [T.RandomResizedCrop(size, scale=(0.75, 1.0)), T.RandomHorizontalFlip()]
        if train else [T.Resize(round(size * 256 / 224)), T.CenterCrop(size)]
    )
    return T.Compose(operations + [T.ToTensor(), T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )])


class ActionDataset(Dataset):
    def __init__(self, manifest, root, classes, size=224, train=False):
        self.rows = pd.read_csv(manifest).to_dict("records")
        self.root = Path(root)
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.transform = image_transforms(size, train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, self.class_to_idx[row["label"]], row["path"]
