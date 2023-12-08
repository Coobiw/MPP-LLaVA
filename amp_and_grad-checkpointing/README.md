# 混合精度技术与gradient checkpointing技术

## 博客

[知乎](https://zhuanlan.zhihu.com/p/671165275?)

## apex库的安装

在使用minigpt4qwen的python环境基础上

下载[apex-23.05版本](https://github.com/NVIDIA/apex/archive/refs/tags/23.05.zip)
解压，然后运行以下命令
```
cd apex-23.05
pip install --upgrade 'pip>=23.1'
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

## 运行示例程序
以下两个程序需要以module的形式运行，其他直接运行即可
```
cd amp_and_grad-checkpointing
python -m apex_amp.test_apex_amp
python -m grad_checkpoint.test_grad_checkpointing
```