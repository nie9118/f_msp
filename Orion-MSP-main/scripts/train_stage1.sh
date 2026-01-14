#!/bin/bash

# This script is used for Stage 1 training of Orion-BiX with rowmixer_lite_icl model

torchrun --nproc_per_node=8 /vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/f_msp/Orion-MSP-main/src/orion_msp/train/runMantis_rowicl.py \
            --model rowmixer_lite_icl \
            --device cuda \
            --amp true \
            --dtype float32 \
            --max_steps 50000 \
            --batch_size 512 \
            --micro_batch_size 4 \
            --lr 1.5e-4 \
            --scheduler cosine_warmup \
            --warmup_proportion 0.01 \
            --gradient_clipping 1.0 \
            --checkpoint_dir ./checkpoints/rowmixer_lite_icl \
            --max_classes 10 \
            --prior_type mix_scm \
            --prior_device cpu \
            --min_features 2 \
            --max_features 100 \
            --max_seq_len 4096 \
            --min_train_size 0.5 \
            --max_train_size 0.8 \
            --embed_dim 128 \
            --row_num_cls 4 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_global 2 \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --perc_num_latents 16 \
            --perc_layers 2 \
            --ff_factor 2 \
            --dropout 0.0 \
            --activation gelu \
            --norm_first true \
            --rowmixer_patch_size 8 \
            --rowmixer_shuffle_p 0.25 \
            --save_temp_every 50 \
            --save_perm_every 1000 \
            --wandb_log False