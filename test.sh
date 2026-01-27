#!/bin/bash
set -euo pipefail  # 开启严格模式，遇到错误立即退出，捕获未定义变量，管道错误传递

# ======================== 配置区 (请根据实际情况修改) ========================
# 要遍历的RowMixer ckpt文件根目录（核心修改：替换的是这个目录下的ckpt）
ROW_MIXER_CKPT_ROOT="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/f_msp/Orion-MSP-main/checkpoints/rowmixer_lite_iclStage2"
# 固定的Mantis checkpoint路径
MANTIS_CKPT="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/test-TIC/TIC-FS/code/checkpoints/CaukerImpro-data100k_emb512_100epochs.pt"
# 评估脚本路径
EVAL_SCRIPT="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/test-TIC/TIC-FS/code/scripts/eval_mantis_rowmixerIclClassifier_ucr_uea.py"
# 数据集路径
UCR_PATH="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/test-TIC/TIC-FS/code/dataset/UCRdata/"
UEA_PATH="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/test-TIC/TIC-FS/code/dataset/UEAData/"
# 可用的GPU卡号 (8卡，可根据实际情况调整)
GPU_IDS=(0 1 2 3 4 5 6 7)
# 每个GPU并行运行的任务数 (根据GPU显存调整，默认1)
TASKS_PER_GPU=1
# 日志输出目录
LOG_DIR="./rowmixer_logs_1"
# ===============================================================================

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 定义执行单个RowMixer ckpt测试的函数
run_rowmixer_eval() {
    local rowmixer_ckpt=$1
    local gpu_id=$2
    # 获取不带后缀的ckpt名称（处理step-13000.ckpt这类命名）
    local ckpt_name=$(basename "${rowmixer_ckpt}" | sed 's/\.[^.]*$//')
    local log_file="${LOG_DIR}/${ckpt_name}_gpu${gpu_id}.log"

    echo "开始执行: ${rowmixer_ckpt} (GPU: ${gpu_id}) | 日志文件: ${log_file}"

    # 设置环境变量并执行评估脚本（核心：替换的是rowmixer-ckpt参数）
    MIOPEN_DISABLE_CACHE=1 \
    MIOPEN_DB_OFF=1 \
    MIOPEN_DEBUG_DISABLE_DB=1 \
    HIP_VISIBLE_DEVICES=${gpu_id} \
    python "${EVAL_SCRIPT}" \
        --suite uea \
        --ucr-path "${UCR_PATH}" \
        --uea-path "${UEA_PATH}" \
        --rowmixer-ckpt "${rowmixer_ckpt}" \
        --mantis-batch-size 16 \
        --rowmixer-batch-size 1 \
        --mantis-ckpt "${MANTIS_CKPT}" > "${log_file}" 2>&1

    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ 完成: ${rowmixer_ckpt} (GPU: ${gpu_id})"
    else
        echo "❌ 失败: ${rowmixer_ckpt} (GPU: ${gpu_id}) | 查看日志: ${log_file}"
    fi
}

# 查找所有RowMixer ckpt文件 (支持.ckpt后缀，可根据需要扩展为.pt等)
ROW_MIXER_CKPT_FILES=($(find "${ROW_MIXER_CKPT_ROOT}" -type f -name "*.ckpt"))

# 检查是否找到ckpt文件
if [ ${#ROW_MIXER_CKPT_FILES[@]} -eq 0 ]; then
    echo "错误: 在目录 ${ROW_MIXER_CKPT_ROOT} 下未找到任何RowMixer ckpt文件！"
    exit 1
fi

echo "找到 ${#ROW_MIXER_CKPT_FILES[@]} 个RowMixer ckpt文件待测试"
echo "使用GPU: ${GPU_IDS[*]} | 每个GPU并行任务数: ${TASKS_PER_GPU}"
echo "日志输出目录: ${LOG_DIR}"
echo "固定Mantis ckpt: ${MANTIS_CKPT}"
echo "================================================"

# 初始化任务计数器和GPU索引
task_counter=0
gpu_index=0

# 遍历所有RowMixer ckpt文件并分配GPU并行执行
for rowmixer_ckpt in "${ROW_MIXER_CKPT_FILES[@]}"; do
    # 获取当前要使用的GPU ID
    current_gpu=${GPU_IDS[$gpu_index]}

    # 后台执行任务
    run_rowmixer_eval "${rowmixer_ckpt}" "${current_gpu}" &

    # 递增计数器
    task_counter=$((task_counter + 1))

    # 计算下一个GPU索引 (循环使用GPU)
    gpu_index=$(((task_counter / TASKS_PER_GPU) % ${#GPU_IDS[@]}))

    # 限制总并行任务数，避免系统负载过高（8卡*1任务=8个并行）
    if [ $((task_counter % (${#GPU_IDS[@]} * TASKS_PER_GPU))) -eq 0 ]; then
        echo "等待当前8卡并行任务批次完成..."
        wait  # 等待所有后台任务完成
    fi
done

# 等待所有剩余的后台任务完成
echo "等待所有RowMixer ckpt测试任务完成..."
wait

echo "================================================"
echo "所有RowMixer ckpt测试完成！"
echo "日志文件位置: ${LOG_DIR}"