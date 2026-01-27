#!/bin/bash
set -euo pipefail  # 开启严格模式，遇到错误立即退出，捕获未定义变量，管道错误传递

# ======================== 配置区 (请根据实际情况修改) ========================
# 要遍历的ckpt文件根目录
CKPT_ROOT_DIR="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/f_msp/Orion-MSP-main/checkpoints/rowmixer_lite_iclStage2/"
# 固定的RowMixer checkpoint路径
ROW_MIXER_CKPT="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/f_msp/Orion-MSP-main/checkpoints/rowmixer_lite_iclStage2/step-13000.ckpt"
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
LOG_DIR="./eval_logs"
# ===============================================================================

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 定义执行单个ckpt测试的函数
run_ckpt_eval() {
    local ckpt_path=$1
    local gpu_id=$2
    local ckpt_name=$(basename "${ckpt_path}" | sed 's/\.[^.]*$//')  # 获取不带后缀的ckpt名称
    local log_file="${LOG_DIR}/${ckpt_name}_gpu${gpu_id}.log"

    echo "开始执行: ${ckpt_path} (GPU: ${gpu_id}) | 日志文件: ${log_file}"

    # 设置环境变量并执行评估脚本
    MIOPEN_DISABLE_CACHE=1 \
    MIOPEN_DB_OFF=1 \
    MIOPEN_DEBUG_DISABLE_DB=1 \
    HIP_VISIBLE_DEVICES=${gpu_id} \
    python "${EVAL_SCRIPT}" \
        --suite uea \
        --ucr-path "${UCR_PATH}" \
        --uea-path "${UEA_PATH}" \
        --rowmixer-ckpt "${ROW_MIXER_CKPT}" \
        --mantis-batch-size 16 \
        --rowmixer-batch-size 1 \
        --mantis-ckpt "${ckpt_path}" > "${log_file}" 2>&1

    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ 完成: ${ckpt_path} (GPU: ${gpu_id})"
    else
        echo "❌ 失败: ${ckpt_path} (GPU: ${gpu_id}) | 查看日志: ${log_file}"
    fi
}

# 查找所有ckpt文件 (支持.pt/.ckpt等后缀，可根据需要扩展)
CKPT_FILES=($(find "${CKPT_ROOT_DIR}" -type f \( -name "*.ckpt" -o -name "*.pt" \)))

# 检查是否找到ckpt文件
if [ ${#CKPT_FILES[@]} -eq 0 ]; then
    echo "错误: 在目录 ${CKPT_ROOT_DIR} 下未找到任何ckpt文件！"
    exit 1
fi

echo "找到 ${#CKPT_FILES[@]} 个ckpt文件待测试"
echo "使用GPU: ${GPU_IDS[*]} | 每个GPU并行任务数: ${TASKS_PER_GPU}"
echo "日志输出目录: ${LOG_DIR}"
echo "================================================"

# 初始化任务计数器和GPU索引
task_counter=0
gpu_index=0

# 遍历所有ckpt文件并分配GPU并行执行
for ckpt_file in "${CKPT_FILES[@]}"; do
    # 获取当前要使用的GPU ID
    current_gpu=${GPU_IDS[$gpu_index]}

    # 后台执行任务
    run_ckpt_eval "${ckpt_file}" "${current_gpu}" &

    # 递增计数器
    task_counter=$((task_counter + 1))

    # 计算下一个GPU索引 (循环使用GPU)
    gpu_index=$(((task_counter / TASKS_PER_GPU) % ${#GPU_IDS[@]}))

    # 可选：限制总并行任务数，避免系统负载过高
    if [ $((task_counter % (${#GPU_IDS[@]} * TASKS_PER_GPU))) -eq 0 ]; then
        echo "等待当前批次任务完成..."
        wait  # 等待所有后台任务完成
    fi
done

# 等待所有剩余的后台任务完成
echo "等待所有测试任务完成..."
wait

echo "================================================"
echo "所有ckpt文件测试完成！"
echo "日志文件位置: ${LOG_DIR}"