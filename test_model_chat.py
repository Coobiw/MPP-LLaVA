import torch
from lavis.models import load_model_and_preprocess

import os
from pathlib import Path

from functools import partial
from PIL import Image
from transformers.generation import GenerationConfig

device = 'cuda'
load_model_and_preprocess = partial(load_model_and_preprocess,is_eval=True,device=device)

ckpt_path = 'lavis/output/instruction_tuning/lr1e-4/20231024110/checkpoint_9.pth'

img_path = 'examples/minigpt4_image_3.jpg'
image = Image.open(img_path).convert('RGB')
text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<Img><ImageHere></Img> Describe this image in detail.<|im_end|>\n<|im_start|>assistant'
model, vis_processors, txt_processors = load_model_and_preprocess("minigpt4qwen", "qwen7b_chat")

model.load_checkpoint(ckpt_path)

sample = {
    'image': vis_processors['eval'](image).unsqueeze(dim=0).cuda(),
    'text': text,
}

generation_config = {
    "chat_format": "chatml",
    "eos_token_id": 151643,
    "pad_token_id": 151643,
    "max_window_size": 6144,
    "max_new_tokens": 512,
    "do_sample": False,
    "transformers_version": "4.31.0"
}

generation_config = GenerationConfig.from_dict(generation_config)

def test_generate():
    print(model.generate(sample,generation_config=generation_config))

def test_chat():
    print(model.chat(query='<Img><ImageHere></Img> Describe this image in detail.',
            history=[],
            image_tensor=sample['image'],
            generation_config=generation_config))

def test_multi_turn():
    response,history = model.chat(query='<Img><ImageHere></Img> Describe this image in detail.',
            history=[],
            image_tensor=sample['image'],
            generation_config=generation_config)
    response,history = model.chat(query='Is there a refrigerator in the picture? Answer yes or no.',
            history=history,
            image_tensor=sample['image'],
            generation_config=generation_config)
    print(response)
    print('===='*10)
    print(history)

# test_generate()
# test_chat()
test_multi_turn()