#!/bin/bash
# CycleGANAS完整运行脚本
# 依次执行：架构搜索 -> 模型训练 -> 测试 -> 关闭实例

# 设置错误处理
set -e

# 配置参数
DATA_DIR="./datasets/horse2zebra"
NAME="horse2zebra"
MODEL="cycle_ganas"
GPU_IDS="0"
BATCH_SIZE="6"
NUM_THREADS="16"
N_EPOCHS="100"
N_EPOCHS_DECAY="100"
SAVE_EPOCH_FREQ="5"
SEARCH_ARCH_EPOCH="200"

# 记录开始时间
START_TIME=$(date)
echo "=============================================="
echo "CycleGANAS训练开始于: $START_TIME"
echo "=============================================="

# 1. 进入工作目录
cd /autodl-tmp/CycleGANAS
echo "当前工作目录: $(pwd)"

# 2. 安装依赖
echo "=============================================="
echo "步骤0: 安装依赖..."
echo "=============================================="
pip install -r requirements.txt

# 3. 运行架构搜索
echo "=============================================="
echo "步骤1: 开始架构搜索..."
echo "=============================================="
python search.py \
    --dataroot $DATA_DIR \
    --name ${NAME}_search \
    --model $MODEL \
    --gpu_ids $GPU_IDS \
    --batch_size $BATCH_SIZE \
    --num_threads $NUM_THREADS \
    --n_epochs $N_EPOCHS \
    --n_epochs_decay $N_EPOCHS_DECAY \
    --save_epoch_freq $SAVE_EPOCH_FREQ

# 4. 运行模型训练
echo "=============================================="
echo "步骤2: 开始模型训练..."
echo "=============================================="
python train.py \
    --dataroot $DATA_DIR \
    --name ${NAME}_train \
    --model $MODEL \
    --arch_dir ./ckpts/${NAME}_search/archs \
    --arch_epoch $SEARCH_ARCH_EPOCH \
    --gpu_ids $GPU_IDS \
    --batch_size $BATCH_SIZE \
    --num_threads $NUM_THREADS \
    --n_epochs $N_EPOCHS \
    --n_epochs_decay $N_EPOCHS_DECAY \
    --save_epoch_freq $SAVE_EPOCH_FREQ

# 5. 运行测试
echo "=============================================="
echo "步骤3: 开始测试..."
echo "=============================================="
python test.py \
    --dataroot $DATA_DIR \
    --name ${NAME}_test \
    --model $MODEL \
    --gpu_ids $GPU_IDS \
    --epoch latest

# 6. 运行评估
echo "=============================================="
echo "步骤4: 开始评估..."
echo "=============================================="
python evaluate.py \
    --dataroot $DATA_DIR \
    --name ${NAME}_eval \
    --model $MODEL \
    --gpu_ids $GPU_IDS

# 记录结束时间
END_TIME=$(date)
echo "=============================================="
echo "CycleGANAS训练完成于: $END_TIME"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "=============================================="

# 7. 关闭实例
echo "=============================================="
echo "训练全部完成，即将关闭实例..."
echo "=============================================="
sleep 10

# 使用AutoDL的关闭命令
# 注意：AutoDL中通常使用以下方式关闭实例
# 方式1：直接关机
echo "正在关闭实例..."
sudo shutdown -h now

# 或者方式2：使用AutoDL特定命令（如果可用）
# autodl stop
