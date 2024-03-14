import json
from transformers import AutoTokenizer

import matplotlib.pyplot as plt
from tqdm import tqdm

llm_tokenizer = AutoTokenizer.from_pretrained(
        "cache/ckpt/Qwen-14B-Chat",
        cache_dir="cache",
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
llm_tokenizer.pad_token_id = llm_tokenizer.eod_id

with open("cache/dataset/llava_pretrain/blip_laion_cc_sbu_558k/llava_pretrain_minigpt4qwen_format.json",'r') as f:
    pretrain_data = json.load(f)
with open("cache/dataset/llava_instruct/llava_instruction_100k.json",'r') as f:
    sft_data = json.load(f)


token_nums = []
im_start = "<|im_start|>"
im_end = "<|im_end|>"

plot_title = ["Pretrain", "SFT"]
num_image = 32
for i,datas in enumerate([pretrain_data, sft_data]):
    token_nums = []
    for data in tqdm(datas):
        question = data["instruction"].replace("<Img><ImageHere></Img> ","")
        answer = data["output"]
        system_message = im_start + "system\nYou are a helpful assistant." + im_end + "\n"
        user_message = im_start + f"user\n{question}" + im_end + "\n"
        assistant_message = im_start + f"assistant\n{answer}" + im_end + "\n"
        whole_text = system_message + user_message + assistant_message

        token_nums.append(len(llm_tokenizer(whole_text).input_ids) + num_image)
    
    plt.hist(token_nums, bins=20, edgecolor='black')
    plt.title(f'Token Lengths Histogram of {plot_title[i]}')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.savefig(f"./vis/{plot_title[i]}_token_distribution.png")

    plt.close()

    print(f"Max Tokens in {plot_title[i]} Stage:\t{max(token_nums)}")

