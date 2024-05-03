import torch
from PIL import Image
from pathlib import Path


class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_files = sorted()
        self.label_files = sorted(os.listdir(os.path.join(data_dir, "labels")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, "images", self.image_files[index])
        label_path = os.path.join(self.data_dir, "labels", self.label_files[index])

        image = Image.open(image_path).convert("RGB")
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")
            labels = [list(map(float, label.split())) for label in labels]
            labels = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels
