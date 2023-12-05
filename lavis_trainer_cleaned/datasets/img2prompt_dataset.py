from common.registry import registry
from .base_dataset import BaseDataset

from pathlib import Path

import clip
import torch

from PIL import Image
import json
import pandas as pd

# dataloader 的 collator
# 因为prompt是字符串，没法拼接成tensor
class Image2PromptCollator:
    def __init__(self,clip_model='./ckpt/clip/openai/RN50.pt'):
        self.txt_model = clip.load(clip_model, device="cpu")[0]

    def __call__(self, batch):
        images, prompts = zip(*batch)
        images = torch.stack(images)

        prompts = clip.tokenize(prompts,context_length=77,truncate=True)
        with torch.no_grad():
            prompt_embeddings = self.txt_model.encode_text(prompts).float()
            prompt_embeddings /= prompt_embeddings.norm(dim=-1,keepdim=True)
        return {"images": images, "prompt_embeddings": prompt_embeddings}

@registry.register_dataset('image2prompt_dataset')
class Image2PromptDataset(BaseDataset):
    def __init__(self, transform, data_path):
        super(Image2PromptDataset,self).__init__(transform)

        self.data_path = Path(data_path)
        if self.data_path.suffix == '.json':
            self.file_type = 'json'
            self.data = json.load(open(self.data_path,'r',encoding='utf-8'))
        elif self.data_path.suffix == '.csv':
            self.file_type = 'csv'
            self.data = pd.read_csv(self.data_path)
        else:
            assert False, 'Suffix of file must be ".csv" or ".json"'

        self.collator = Image2PromptCollator()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.file_type == 'csv':
            row = self.data.iloc[idx]
            image = Image.open(row['image_path']).convert("RGB")
            image = self.transform(image)
            prompt = row['prompt']
        else:
            item = self.data[idx]
            image = Image.open(item['image_path']).convert("RGB")
            image = self.transform(image)
            prompt = item['prompt']
        return image, prompt
