# Model Weights Preparation

## 准备好后的完整目录结构

```bash
├── cache
│   ├── ckpt
│   │   ├── bert-base-uncased
│   │   ├── blip2
│   │   │   ├── blip2_pretrained_flant5xxl.pth
│   │   ├── eva
│   │   │   ├── eva_vit_g.pth
│   │   ├── Qwen7B-chat
│   │   │   ├── config.json
│   │   │   ├── ...
├── ...
├── lavis
│   ├── output
│   │   ├── pp_7b_video
│   │   │   ├── pretrain
│   │   │   |   ├── global_step2181
│   │   │   |   |   ├── model.pth
│   │   │   ├── sft_video
│   │   │   |   ├── global_step2005
│   │   │   |   |   ├── unfreeze_llm_model.pth
```

## 准备过程
> 请将模型权重下载后都放在 `cache/ckpt`下

```bash
mkdir cache
cd cache
mkdir ckpt
mkdir dataset
```

### 1.下载BLIP2的相关权重

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

### 2.下载Qwen7B-Chat的权重

[Qwen-7B-chat huggingface](https://huggingface.co/Qwen/Qwen-7B-Chat)

### 3.获得一阶段pretrain后的checkpoint（optional，如果你想直接在这上面做sft的话）

(建议放入 `lavis/output/pp_7b_video/pretrain/global_step2181`)

在本仓库的release里放有checkpoint，可以直接下载

```bash
wget https://github.com/Coobiw/MiniGPT4Qwen/releases/download/MPP-Qwen-Next_ckpt-and-data/ckpt-and-data.zip
unzip ckpt-and-data.zip
```

### 4.sft后的权重（百度网盘/modelscope）
- [modelscope链接](https://www.modelscope.cn/models/Coobiw/MPP-Qwen-Next)
- [百度网盘链接](https://pan.baidu.com/s/15rfwuCfM_sdViWQJv1mZmg?pwd=baka)