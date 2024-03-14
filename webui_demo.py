import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image

from transformers.generation import GenerationConfig

from lavis.common.config import Config
from lavis.common.dist_utils import get_rank
from lavis.common.registry import registry
from lavis.models import load_model_and_preprocess

from functools import partial
from copy import deepcopy

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def _load_model_processor(args):
    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = 'cuda:{}'.format(args.gpu_id)
    
    global load_model_and_preprocess
    load_model_and_preprocess = partial(load_model_and_preprocess,is_eval=True,device=device_map)

    model, vis_processors, _ = load_model_and_preprocess("minigpt4qwen", args.model_type)
    model.load_checkpoint(args.checkpoint_path)

    model.llm_model.transformer.bfloat16()
    model.llm_model.lm_head.bfloat16()

    generation_config = {
    "chat_format": "chatml",
    "eos_token_id": 151643,
    "pad_token_id": 151643,
    "max_window_size": 6144,
    "max_new_tokens": 512,
    "transformers_version": "4.31.0"
    }

    return model, vis_processors, generation_config

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-type",type=str,default='qwen7b_chat',choices=['qwen7b_chat','qwen14b_chat'])
    parser.add_argument("-c", "--checkpoint-path", type=str,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    args = parser.parse_args()
    return args

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()

if torch.cuda.is_available() and not args.cpu_only:
    device='cuda:{}'.format(args.gpu_id)
else:
    device=torch.device('cpu')

disable_torch_init()
model, vis_processors, default_generation_config = _load_model_processor(args)
vis_processor = vis_processors["eval"]

print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================


def gradio_reset(history, img_list):
    if history is not None:
        history = []
    if img_list is not None:
        img_list = []
    return None, \
        gr.update(value=None, interactive=True), \
        gr.update(placeholder='Please upload your image first', interactive=False),\
        gr.update(value="Upload & Start Chat", interactive=True), \
        history, \
        img_list

def upload_img(gr_img, text_input, history, img_list, img_prefix):
    def load_and_process_img(image,img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = vis_processor(raw_image).unsqueeze(0).to(device)
        elif isinstance(image, Image.Image):
            raw_image = image
            raw_image = raw_image.convert('RGB')
            image = vis_processor(raw_image).unsqueeze(0).to(device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(device)

        img_list.append(image)
        msg = "Received."
        return msg
    if gr_img is None:
        return None, None, gr.update(interactive=True), history, None

    llm_message = load_and_process_img(gr_img, img_list)
    img_prefix = '<Img><ImageHere></Img>'
    return gr.update(interactive=False), \
        gr.update(interactive=True, placeholder='Type and press Enter'), \
        gr.update(value="Start Chatting", interactive=False), \
        history, \
        img_list, \
        img_prefix

def gradio_ask(user_message, chatbot, img_prefix):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, history
    def get_ask(user_message, img_prefix):
        return img_prefix + user_message
    user_message = get_ask(user_message,img_prefix)
    chatbot = chatbot + [[user_message, None]]
    img_prefix = ""
    return '', chatbot, img_prefix


def gradio_answer(chatbot, history, img_list, do_sample,num_beams, temperature, top_k, top_p):
    generation_config = deepcopy(default_generation_config)
    generation_config.update(
        {
            "do_sample": do_sample=='True',
            "num_beams": num_beams,
            'temperature': temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
    )
    image_tensor =  img_list[0]  # 如果想支持多图情况：torch.stack(img_list).to(self.device)
    generation_config = GenerationConfig.from_dict(generation_config)
    global args
    if args.cpu_only:
        model.bfloat16()
        response, history = model.chat(query=chatbot[-1][0], history=history, image_tensor=image_tensor.bfloat16(), generation_config=generation_config,verbose=True)
    else:
        with torch.cuda.amp.autocast(enabled=True,dtype=torch.bfloat16):
            response, history = model.chat(query=chatbot[-1][0], history=history, image_tensor=image_tensor.bfloat16(), generation_config=generation_config,verbose=True)
    chatbot[-1][1] = response
    return chatbot, history, img_list

title = """<h1 align="center">Demo of MiniGPT4Qwen</h1>"""
description = """<h3>This is the demo of MiniGPT4Qwen. Upload your images and start chatting! <br> To use
            example questions, click example image, hit upload, and press enter in the chatbox. </h3>"""

from transformers.trainer_utils import set_seed
set_seed(args.seed)
#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart 🔄")
            do_sample = gr.components.Radio(['True', 'False'],
                            label='do_sample(If False, num_beams, temperature and so on cannot work!)',
                            value='False')

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            top_k = gr.Slider(
                minimum=0,
                maximum=5,
                value=1,
                step=1,
                interactive=True,
                label="Top_k",
            )

            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=1.0,
                step=0.05,
                interactive=True,
                label="Top_p",
            )

        with gr.Column():
            history = gr.State(value=[])
            img_list = gr.State(value=[])
            chatbot = gr.Chatbot(label='MiniGPT4Qwen')
            img_prefix = gr.State(value="")
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)

            gr.Examples(examples=[
                ["examples/minigpt4_image_3.jpg", "描述下这幅图片"],
            ], inputs=[image, text_input])

    upload_button.click(upload_img, [image, text_input, history,img_list, img_prefix], [image, text_input, upload_button, history, img_list, img_prefix])

    print(list(map(type,[text_input, chatbot, img_prefix])))
    print(list(map(type,[chatbot, history, img_list, do_sample, num_beams, temperature, top_k, top_p])))
    text_input.submit(gradio_ask, [text_input, chatbot, img_prefix], [text_input, chatbot, img_prefix]).then(
        gradio_answer, [chatbot, history, img_list, do_sample, num_beams, temperature, top_k, top_p], [chatbot, history, img_list]
    )
    clear.click(gradio_reset, [history, img_list], [chatbot, image, text_input, upload_button, history, img_list], queue=False)

demo.launch(share=True,inbrowser=True)