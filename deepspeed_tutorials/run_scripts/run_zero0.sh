python -m torch.distributed.run \
    --nproc_per_node=8 \
    test_zero_no_optim.py \
    --deepspeed_config ds_configs/ds_config_zero0.json \