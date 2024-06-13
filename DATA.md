# Training Data Preparation
如果你只需要进行推理和测试，则无需准备训练数据

下载后按照下面的目录结构组织文件

最后需要将数据集放入 `./cache/dataset`中，目录结构如下：

```bash
├── cache
│   |── dataset
│   |   ├── llava_pretrain
│   │   │   ├── blip_laion_cc_sbu_558k
│   │   │   |   ├── images
│   │   │   |   ├── llava_pretrain_minigpt4qwen_format.json
│   |   ├── llava_instuct
│   │   │   ├── coco
│   │   │   |   ├── train2017
│   │   │   ├── llava_instruction_156k.json # 包含了多轮对话的sft数据
│   |   ├── videochatgpt
│   │   │   ├── videochatgpt_tune # 包含视频文件
│   │   │   |   ├── xxx.mp4
│   │   │   |   ├── xxx.mkv
│   │   │   |   ├── ...
│   │   │   ├── videochatgpt_instruction_100k.json
```
## LLaVA Data
MPP-Qwen-Next使用了LLaVA的Pretrain和SFT的数据集，这部分整体数据获取流程与LLaVA仓库说明的大体一致。

预训练数据：[558K subset of the LAION-CC-SBU dataset with BLIP captions](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)，去该huggingface链接下载`images.zip`和`blip_laion_cc_sbu_558k.json`

指令微调数据：下载coco的train2017里的图片：
```bash
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```

MPP-Qwen-Next format的标注json文件：在本仓库的release中([https://github.com/Coobiw/MiniGPT4Qwen/releases/tag/MPP-Qwen-Next_ckpt-and-data](https://github.com/Coobiw/MiniGPT4Qwen/releases/tag/MPP-Qwen-Next_ckpt-and-data)):
```bash
wget https://github.com/Coobiw/MiniGPT4Qwen/releases/download/MPP-Qwen-Next_ckpt-and-data/ckpt-and-data.zip
unzip ckpt-and-data.zip
```

## VideoChatGPT SFT DATA

### 视频数据

#### 百度网盘

参考[VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md)的videochatgpt获取tutorials，其提供的百度网盘提取链接如下：https://pan.baidu.com/share/init?surl=0hJ_U7wVmYTUo75YHc_n8g&pwd=g1hf

#### Google Drive/OneDrive(VideoChatGPT官方提供)

请根据其github issue：https://github.com/mbzuai-oryx/Video-ChatGPT/issues/98，进行下载

### 处理好的annotations json文件
MPP-Qwen-Next format的标注json文件：在本仓库的release中([https://github.com/Coobiw/MiniGPT4Qwen/releases/tag/MPP-Qwen-Next_ckpt-and-data](https://github.com/Coobiw/MiniGPT4Qwen/releases/tag/MPP-Qwen-Next_ckpt-and-data)):
```bash
wget https://github.com/Coobiw/MiniGPT4Qwen/releases/download/MPP-Qwen-Next_ckpt-and-data/ckpt-and-data.zip
unzip ckpt-and-data.zip
```

内含`videochatgpt_instruction_100k.json`