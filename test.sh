#!/bin/bash
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_DB_OFF=1
export MIOPEN_DEBUG_DISABLE_DB=1
export MIOPEN_USER_DB_PATH=:memory:
export MIOPEN_SYSTEM_DB_PATH=:memory:
export HIP_VISIBLE_DEVICES=0

ROW_MIXER_CKPT="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/f_msp/Orion-MSP-main/checkpoints/rowmixer_lite_iclStage2/step-15600.ckpt"

# mantis 模型 checkpoint 文件的实际路径（必填，替换下面的占位符）
MANTIS_CKPT="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/test-TIC/TIC-FS/code/checkpoints/CaukerImpro-data100k_emb512_100epochs.pt"

# 固定路径（无需修改，根据你的原始命令填充）
EVAL_SCRIPT="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/test-TIC/TIC-FS/code/scripts/eval_mantis_rowmixerIclClassifier_ucr_uea.py"
UCR_PATH="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/test-TIC/TIC-FS/code/dataset/UCRdata/"
UEA_PATH="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/test-TIC/TIC-FS/code/dataset/UEAData/"


python "${EVAL_SCRIPT}" \
    --suite uea \
    --ucr-path "${UCR_PATH}" \
    --uea-path "${UEA_PATH}" \
    --rowmixer-ckpt "${ROW_MIXER_CKPT}" \
    --use-var-selector \
    --var-num-channels 10 \
    --mantis-batch-size 512 \
    --rowmixer-batch-size 64 \
    --mantis-ckpt "${MANTIS_CKPT}"