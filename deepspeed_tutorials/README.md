# 关于DeepSpeed的尝试

知乎博客地址： https://zhuanlan.zhihu.com/p/673359684



## 参考

Repo：https://github.com/microsoft/DeepSpeedExamples

https://github.com/microsoft/DeepSpeedExamples/blob/master/training/HelloDeepSpeed/train_bert_ds.py，代码拷贝到了本项目的：https://github.com/Coobiw/MiniGPT4Qwen/blob/master/deepspeed_tutorials/train_bert_official.py



## EVA-ViT-G（1B）上的实验

| Setting（bf16，不开gradient-checkpointing） | max_allocated_memory/GB（若无说明则bs=32） | time per epoch/s（bs=32 8卡 共400条数据） |
| ------------------------------------------- | ------------------------------------------ | ----------------------------------------- |
| ZERO-0 (DDP)                                | bs=32 OOM bs=16 18.36                      | 11.57                                     |
| ZERO-1                                      | 21.46                                      | 9.68                                      |
| ZERO-1 (offload optimizer)                  | 20.40                                      | 13.45                                     |
| ZERO-2                                      | 22.26                                      | 8.51                                      |
| ZERO-2 (offload optimizer)                  | 20.80                                      | 13.34                                     |
| ZERO-3                                      | 22.37                                      | 7.94                                      |
| ZERO-3 (offload optimizer)                  | 20.39                                      | 12.67                                     |
| ZERO-3 (offload optimizer + params)         | 20.39                                      | 12.08                                     |

