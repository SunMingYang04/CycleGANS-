# CycleGANAS自动训练脚本使用说明

## 概述

这个脚本可以自动完成CycleGANAS的完整训练流程：
1. 架构搜索
2. 模型训练
3. 测试
4. 自动关闭实例

## 使用方法

### 1. 将脚本上传到AutoDL实例

将 `run_cycleganas.sh` 上传到你的AutoDL实例的 `/autodl-tmp/` 目录下

### 2. 激活conda环境

```bash
conda activate cycleganas
```

### 3. 运行脚本

```bash
cd /autodl-tmp
bash run_cycleganas.sh
```

或者使用后台运行：

```bash
cd /autodl-tmp
nohup bash run_cycleganas.sh > training.log 2>&1 &
```

### 4. 查看日志

如果使用后台运行，可以通过以下命令查看日志：

```bash
tail -f training.log
```

## 脚本功能说明

### 参数配置

脚本顶部的参数可以根据需要修改：

```bash
DATA_DIR="./datasets/horse2zebra"  # 数据集路径
NAME="horse2zebra"                 # 实验名称
MODEL="cycle_ganas"               # 模型类型
GPU_IDS="0"                       # GPU ID
BATCH_SIZE="6"                    # 批次大小（根据GPU显存调整）
NUM_THREADS="16"                  # 数据加载线程数
N_EPOCHS="100"                    # 初始学习率的epoch数
N_EPOCHS_DECAY="100"              # 学习率衰减的epoch数
SAVE_EPOCH_FREQ="5"               # 保存检查点的频率
SEARCH_ARCH_EPOCH="200"           # 使用的架构搜索epoch
```

### 执行步骤

1. **架构搜索**：运行 `search.py` 进行神经网络架构搜索
2. **模型训练**：使用搜索到的最佳架构运行 `train.py`
3. **测试**：运行 `test.py` 测试训练好的模型
4. **关闭实例**：全部完成后自动关闭实例以节省费用

## 注意事项

### 1. 前置条件

确保已经完成以下准备工作：
- 数据集已解压到 `./datasets/horse2zebra/`
- 所有依赖已安装（PyTorch, cleanfid等）
- conda环境已激活

### 2. 后台运行建议

由于完整训练可能需要数天时间，建议使用后台运行：

```bash
cd /autodl-tmp/CycleGANAS
nohup bash ../run_cycleganas.sh > training.log 2>&1 &
```

### 3. GPU显存调整

- RTX 5090 (32GB): 可以使用 batch_size=6-8
- RTX 4090 (24GB): 可以使用 batch_size=4-6
- 如遇OOM错误，减小batch_size

### 4. 结果保存位置

- 架构搜索结果：`./ckpts/horse2zebra_search/`
- 训练检查点：`./ckpts/horse2zebra_train/`
- 测试结果：`./results/horse2zebra_test/`

### 5. 安全关闭

脚本最后会自动关闭实例，如果你想在测试后手动检查结果，可以注释掉最后几行关机命令。

## 故障排除

### 脚本无法执行

```bash
chmod +x run_cycleganas.sh
```

### 权限问题

如果关机命令需要权限，可以修改脚本中的关机命令或手动关闭实例。

### 想中途停止训练

```bash
# 查找进程
ps aux | grep python
# 杀死进程
kill <PID>
```

## 快速开始

如果你已经准备好了环境，只需要：

```bash
# 1. 进入目录
cd /autodl-tmp/CycleGANAS

# 2. 后台运行
nohup bash /workspace/run_cycleganas.sh > training.log 2>&1 &

# 3. 查看进度
tail -f training.log
```

就可以开始完整的训练流程了！
