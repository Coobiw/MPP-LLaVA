from common.registry import registry
from .base_dataset import BaseDataset

from PIL import Image
from pathlib import Path
import os

import torch

@registry.register_dataset('example_cls_dataset')
class ExampleClsDataset(BaseDataset):
    def __init__(self, transform, data_root):
        super().__init__(transform=transform)
        self.data_root = Path(data_root)
        self.cls_dir = sorted(list(os.listdir(self.data_root)))

        self.data = []
        self.labels = []

        self.idx2cls,self.cls2idx = {}, {}
        for i,cls in enumerate(self.cls_dir):
            self.idx2cls[i] = cls
            self.cls2idx[cls] = i
            imgs = [str(self.data_root/cls/img) for img in os.listdir(self.data_root/cls) if self.is_img(img)]
            self.data.extend(imgs)
            self.labels.extend([i]*len(imgs))

        assert len(self.data) == len(self.labels)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        return self.transform(image), torch.tensor(label,dtype=torch.long)

    @staticmethod
    def is_img(img_path):
        return Path(img_path).suffix.lower() in ['.jpg', '.jpeg', '.png']

    def collator(self,batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)

        return {"images": images, "labels": labels}