#!/bin/bash

# This script is used for stage 1 training of Orion-MSP

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun  --nproc_per_node=8 /vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/f_msp/Orion-MSP-main/src/orion_msp/train/run.py \
            --wandb_log True \
            --model rowmixer_lite_icl \
            --wandb_project TabICL \
            --wandb_name fang \
            --wandb_dir ./wandb/dir \
            --wandb_mode offline \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 20000 \
            --batch_size 256 \
            --micro_batch_size 1 \
            --lr 0.75e-4 \
            --scheduler cosine_warmup \
            --warmup_proportion 0.01 \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --prior_device cpu \
            --batch_size_per_gp 1 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len 4096 \
            --min_train_size 0.1 \
            --max_train_size 0.9 \
            --embed_dim 96 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 4 \
            --row_num_cls 2 \
            --row_rope_base 100000 \
            --row_scales 1,4 \
            --row_window 4 \
            --row_num_random 2 \
            --row_num_global 2 \
            --row_group_mode pma \
            --perc_num_latents 64 \
            --perc_layers 3 \
            --icl_num_blocks 6 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir ./checkpoints_s2/rowmixer_lite_icl \
            --checkpoint_path ./checkpoints/rowmixer_lite_icl/step-12000.ckpt \
            --save_temp_every 50 \
            --save_perm_every 1000