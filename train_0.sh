#!/bin/bash

# 主节点（rank=0）
python -m torch.distributed.run \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --master_addr=10.88.129.86 \
    --master_port=12345 \
    train.py \
        --load_model 'model/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth' \
        --proj_dir 'output' \
        --data_file 'data/message_cache' \
        --data_type binidx \
        --vocab_size 65536 \
        --ctx_len 4096 \
        --epoch_steps 130000 \
        --epoch_count 10 \
        --epoch_begin 0 \
        --epoch_save 1 \
        --micro_bsz 1 \
        --n_layer 32 \
        --n_embd 2560 \
        --lr_init 1e-5 \
        --lr_final 1e-6 \
        --warmup_steps 200 \
        --beta1 0.9 \
        --beta2 0.99 \
        --adam_eps 1e-6 \
        --accelerator gpu \
        --devices 1 \
        --precision bf16 \
        --strategy deepspeed_stage_1 \
        --num_nodes 2 \
        --grad_cp 1 \
        --accumulate_grad_batches 32 \
        --dataload pad \
        --chunk_ctx 512 \
        --data_shuffle 1 \
        --wandb "RWKV7-ndt" \
        --quant none \
        --peft disha \
        --disha_config '{"mode":"bat","load":"","r":640}' \
        --loss_mask none \
        --op triton \
        --lr_schedule cos \
        --my_testing x070 

