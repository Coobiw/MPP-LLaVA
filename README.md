- [Minigpt4Qwen](#minigpt4qwen)
  - [Introduction](#introduction)
  - [所需计算资源](#所需计算资源)
  - [TODO LIST](#todo-list)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
    - [模型下载](#模型下载)
    - [运行test\_model\_chat.py进行初步尝试](#运行test_model_chatpy进行初步尝试)
    - [运行命令行demo](#运行命令行demo)
  - [训练](#训练)
    - [数据准备](#数据准备)
    - [config文件的书写](#config文件的书写)
    - [运行train.py](#运行trainpy)
  - [Minigpt4Qwen对话示例](#minigpt4qwen对话示例)
  - [Acknowledgement](#acknowledgement)
  - [License](#license)

# Minigpt4Qwen

## Introduction

[MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)是最近很火的一个MLLM项目，他证明了对于BLIP2的ViT+Q-former这种已经与语言模态做了对齐预训练的结构，**只需要重训一个Linear层，便可以接入新的LLM**。对于现在这个每个月有一个新的更强的LLM出来的时代，这种构建多模态大模型的方式是十分高效的。

然而，MiniGPT4采用LLaMA、Vicuna作为语言模型，它们的中文支持相对较弱，导致训练出的MLLM对中文支持不好。而现在也有许多开源出来的中文LLM，如：阿里云的Qwen、百川智能的baichuan等。

本项目使用Qwen-Chat作为LLM，用MiniGPT4的对齐方式，更加高效地训练了一个MLLM，名为 `Minigpt4Qwen`。**相比MiniGPT4的两阶段训练（低质量数据对齐 + 高质量数据指令微调），本项目仅仅采用18.8k的高质量指令微调数据，经过单阶段预训练即可达到很好的效果。**

## 所需计算资源

本项目使用了8张 `3090 24G`，进行训练和推理。**实际上，单张24G的3090也能够满足计算需求。**

## TODO LIST

- [ ] 支持deepspeed
- [ ] 支持pytorch原生FSDP
- [ ] 开放gradio WebUI demo
- [X] 开放所用数据集和checkpoint
- [X] 开放源代码

## Installation

```bash
conda create -n minigpt4qwen python=3.8
conda activate minigpt4qwen
pip install -e .
```

## Getting Started

### 模型下载

> 请将模型权重下载后都放在 `cache/ckpt`下

```bash
mkdir cache
cd cache
mkdir ckpt
mkdir dataset
```

1.下载BLIP2的相关权重

(a) eva vit-g

[eva_vit_g.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)

```bash
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
```

(b) bert-base-uncased

[huggingface](https://huggingface.co/bert-base-uncased/tree/main),下载如下的文件即可

![image-20231026013454256](./assets/image-20231026013454256.png)

(c) blip2_pretrained_flant5xxl

[blip2_pretrained_flant5xxl.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth)

```bash
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth
```

2.下载Qwen7B-chat的权重

[Qwen-7B-chat huggingface](https://huggingface.co/Qwen/Qwen-7B-Chat)

3.下载本模型的checkpoint(建议放入 `lavis/output/`)

在本仓库的release里放有checkpoint，可以直接下载

```bash
wget https://github.com/Coobiw/MiniGPT4Qwen/releases/download/instruction-data_and_checkpointv1.0/ckpt.zip
unzip ckpt.zip
```

目录结构：

```bash
├── cache
│   ├── ckpt
│   │   ├── bert-base-uncased
│   │   ├── blip2
│   │   │   ├── blip2_pretrained_flant5xxl.pth
│   │   ├── eva
│   │   │   ├── eva_vit_g.pth
│   │   ├── Qwen7B-chat
```

### 运行test_model_chat.py进行初步尝试

```bash
python test_model_chat.py
```

你可以修改里面的[ckpt_path](https://github.com/Coobiw/MiniGPT4Qwen/blob/8fe80125b91af858b528d74c13c40fbb2fd90ad5/test_model_chat.py#L14)和[img_path](https://github.com/Coobiw/MiniGPT4Qwen/blob/8fe80125b91af858b528d74c13c40fbb2fd90ad5/test_model_chat.py#L16)

### 运行命令行demo

```bash
python cli_demo.py --checkpoint-path xxxxxx
```

运行后需要输入图片路径，输入后进入对话

常见操作：

> :help 查看help
>
> :clear 清空当前命令行
>
> :clh 清空对话历史（但图像输入不会更改）
>
> :his 查看对话历史
>
> :img 查看输入的图像路径

## 训练

### 数据准备

本数据集共含有18.8k个图文对，来自[MMPretrain](https://github.com/open-mmlab/mmpretrain)根据llava和minigpt4处理得到，下载链接：[huggingface](https://huggingface.co/datasets/deepHug/minigpt4_training_for_MMPretrain)

为了支持当前的 `lavis`库的训练框架，我对数据集的annotations进行了重新处理，放到了本仓库的release中，下载链接：[instruction_data](https://github.com/Coobiw/MiniGPT4Qwen/releases/download/instruction-data_and_checkpointv1.0/instruction_data.zip)

```bash
wget https://github.com/Coobiw/MiniGPT4Qwen/releases/download/instruction-data_and_checkpointv1.0/instruction_data.zip
unzip instruction_data
```

最后需要将数据集放入 `./cache/dataset`中，目录结构如下：

```bash
├── cache
│   └── dataset
│       ├── llava
│   │   │   ├── llava_minigpt4qwen_format.json
│   │   │   ├── image
│       ├── minigpt4
│   │   │   ├── image
│   │   │   ├── minigpt4_minigpt4qwen_format.json
```

### config文件的书写

请参考[train.yaml](https://github.com/Coobiw/MiniGPT4Qwen/blob/master/lavis/projects/instruction_tuning/train.yaml)

### 运行train.py

单卡：

```bash
CUDA_VISIBLE_DEVICES=xxx python train.py --cfg-path lavis/projects/instruction_tuning/train.yaml
```

多卡：

```bash
CUDA_VISIBLE_DEVICES=xxx python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/instruction_tuning/train.yaml
```

## Minigpt4Qwen对话示例

输入图片:

 ![](./examples/minigpt4_image_3.jpg)

对话内容展示：（通过 `:his`)

![image-20231026014107033](./assets/image-20231026014107033.png)

## Acknowledgement

- [Lavis](https://github.com/salesforce/LAVIS) 本仓库是基于lavis进行构建的
- [QwenLM](https://github.com/QwenLM/Qwen) 本仓库的语言模型采用Qwen-7B-Chat
- [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4) 本仓库的主要思想来自MiniGPT4

## License

- 本仓库的许多代码是基于[Lavis](https://github.com/salesforce/LAVIS) 的，其采用 [BSD 3-Clause License](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/LICENSE_Lavis.md).
- 本仓库采用Qwen-7B-Chat，支持商用和科研、开发用途，其License为[LICENSE](https://github.com/QwenLM/Qwen/blob/main/LICENSE)
