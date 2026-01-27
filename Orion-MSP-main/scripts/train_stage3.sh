##!/bin/bash
#
## This script is used for stage 1 training of Orion-MSP
#
## ----------------------------------
## Generate prior datasets on the fly
## ----------------------------------
#
#torchrun --standalone --nproc_per_node=1 /path/to/orion_msp/train/run.py \
#            --wandb_log True \
#            --wandb_project Orion-MSP \
#            --wandb_name Stage3 \
#            --wandb_dir /my/wandb/dir \
#            --wandb_mode online \
#            --device cuda \
#            --dtype float32 \
#            --np_seed 42 \
#            --torch_seed 42 \
#            --max_steps 50 \
#            --batch_size 512 \
#            --micro_batch_size 1 \
#            --lr 2e-6 \
#            --scheduler constant \
#            --gradient_clipping 1.0 \
#            --prior_type mix_scm \
#            --prior_device cpu \
#            --batch_size_per_gp 4 \
#            --min_features 2 \
#            --max_features 100 \
#            --max_classes 10 \
#            --max_seq_len 1024 \
#            --min_train_size 0.1 \
#            --max_train_size 0.9 \
#            --embed_dim 128 \
#            --col_num_blocks 3 \
#            --col_nhead 4 \
#            --col_num_inds 128 \
#            --row_num_blocks 6 \
#            --row_nhead 8 \
#            --row_num_cls 4 \
#            --row_rope_base 100000 \
#            --row_scales 1,4,16 \
#            --row_window 8 \
#            --row_num_random 2 \
#            --row_num_global 8 \
#            --row_group_mode pma \
#            --perc_num_latents 64 \
#            --perc_layers 3 \
#            --icl_num_blocks 12 \
#            --icl_nhead 4 \
#            --ff_factor 2 \
#            --norm_first True \
#            --checkpoint_dir /my/stage1/checkpoint/dir \
#            --save_temp_every 2 \
#            --save_perm_every 10
#
#
#
## Loading prior data from disk and training
#torchrun --standalone --nproc_per_node=1 /path/to/orion_msp/train/run.py \
#            --wandb_log True \
#            --wandb_project Orion-MSP \
#            --wandb_name Stage3 \
#            --wandb_dir /my/wandb/dir \
#            --wandb_mode online \
#            --device cuda \
#            --dtype float32 \
#            --np_seed 42 \
#            --torch_seed 42 \
#            --max_steps 50 \
#            --batch_size 512 \
#            --micro_batch_size 1 \
#            --lr 2e-6 \
#            --scheduler constant \
#            --gradient_clipping 1.0 \
#            --prior_dir /my/stage1/prior/dir \
#            --load_prior_start 0 \
#            --delete_after_load False \
#            --prior_device cpu \
#            --embed_dim 128 \
#            --col_num_blocks 3 \
#            --col_nhead 4 \
#            --col_num_inds 128 \
#            --freeze_col True \
#            --row_num_blocks 6 \
#            --row_nhead 8 \
#            --row_rope_base 100000 \
#            --row_scales 1,4,16 \
#            --row_window 8 \
#            --row_num_random 2 \
#            --row_num_global 4 \
#            --row_num_cls 4 \
#            --row_group_mode pma \
#            --freeze_row True \
#            --perc_num_latents 64 \
#            --perc_layers 3 \
#            --icl_num_blocks 12 \
#            --icl_nhead 4 \
#            --ff_factor 2 \
#            --norm_first True \
#            --checkpoint_dir /my/stage1/checkpoint/dir \
#            --save_temp_every 2 \
#            --save_perm_every 10

# This script is used for Stage 1 training of Orion-BiX with rowmixer_lite_icl model

#offline-run-20260124_175942-pn67q39c
torchrun --nproc_per_node=8 /vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/f_msp/Orion-MSP-main/src/orion_msp/train/run.py \
            --wandb_log True \
            --model rowmixer_lite_icl \
            --wandb_project TabICL \
            --wandb_name fang \
            --wandb_dir ./wandb/dir \
            --wandb_mode offline \
            --model rowmixer_lite_icl \
            --device cuda \
            --amp true \
            --freeze_row True \
            --dtype float32 \
            --max_steps 14200 \
            --batch_size 256 \
            --micro_batch_size 2 \
            --lr 2e-6 \
            --scheduler constant \
            --warmup_proportion 0.01 \
            --gradient_clipping 1.0 \
            --checkpoint_dir ./checkpoints/rowmixer_lite_iclStage3 \
            --max_classes 10 \
            --prior_type mix_scm \
            --prior_device cpu \
            --min_features 2 \
            --max_features 100 \
            --max_seq_len 10000 \
            --min_train_size 0.2 \
            --max_train_size 0.9 \
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
            --checkpoint_path ./checkpoints/rowmixer_lite_iclStage2/step-14000.ckpt \
            --save_temp_every 2 \
            --save_perm_every 10