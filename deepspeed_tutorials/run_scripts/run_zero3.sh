CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port 29500 \
    test_zero_no_optim.py \
    --deepspeed_config ds_configs/ds_config_zero3.json \