# 图像embedding接入LLM方法

- BLIP2的方法的缺陷：image embedding只能当prefix，无法灵活接入句子中的任何位置
- Qwen-VL方法的缺陷：因为需要 `<img>` 和 `</img>`，即 `image_start`和 `image_end`，所以需要引入两个额外的token，因此需要重新训练 `word embedding`层和最后的 `lm_head `
- 这里的方法，使用Qwen-VL的tokenizer里的<|extra_0|> token作为一个**占位符**，后面用image_embedding代替即可

# 遇到的问题

## 分布式训练DDP

- 最开始安装了最新的torch==2.1.0，会出现DDP（python -m torch.distributed.run --nproc_per_node=8 train.py)，出现loss为nan的问题
  - 解决方法，降级为torch==2.0.1即可，具体原因还没有去研究
