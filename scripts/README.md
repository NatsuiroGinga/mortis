# Mortis 预训练脚本使用说明

## 概述

`pretrain.sh` 是一个智能的预训练启动脚本，支持不同显存容量的GPU配置。脚本会自动根据选择的模式调整模型参数和训练配置，以最大化利用可用显存。

## 快速开始

### 基本用法

```bash
# 14G显存模式 (适合RTX 4060 Ti 16G等显卡)
./scripts/pretrain.sh 14g

# 24G显存模式 (适合RTX 3090/4090等显卡)
./scripts/pretrain.sh 24g
```

### 高级用法

```bash
# 14G模式，训练2轮，启用wandb日志
./scripts/pretrain.sh 14g -e 2 -w

# 24G模式，自定义数据路径和保存目录
./scripts/pretrain.sh 24g -d /path/to/data.jsonl -s /path/to/output

# 自定义学习率
./scripts/pretrain.sh 14g -l 1e-3 -e 3
```

## 模式配置详情

### 14G显存模式
- **适用显卡**: RTX 4060 Ti 16G, RTX 3070 16G等
- **模型配置**:
  - 隐藏层维度: 512
  - Transformer层数: 8
  - 最大序列长度: 512
- **训练配置**:
  - Batch Size: 16
  - 梯度累积步数: 4
  - 有效Batch Size: 64

### 24G显存模式
- **适用显卡**: RTX 3090 24G, RTX 4090 24G等
- **模型配置**:
  - 隐藏层维度: 768
  - Transformer层数: 16
  - 最大序列长度: 1024
- **训练配置**:
  - Batch Size: 32
  - 梯度累积步数: 2
  - 有效Batch Size: 64

## 参数说明

### 必需参数
- `模式` (14g/24g): 选择显存模式

### 可选参数
- `-d, --data-path PATH`: 数据文件路径 (默认: ../dataset/pretrain_hq.jsonl)
- `-e, --epochs NUM`: 训练轮数 (默认: 1)
- `-l, --learning-rate LR`: 学习率 (默认: 5e-4)
- `-s, --save-dir DIR`: 模型保存目录 (默认: ../out)
- `-w, --wandb`: 启用wandb日志记录
- `-h, --help`: 显示帮助信息

## 环境要求

### 硬件要求
- **GPU**: NVIDIA显卡，支持CUDA
- **显存**: 至少14GB可用显存
- **内存**: 建议32GB以上

### 软件要求
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.7+
- **依赖包**: 见项目根目录requirements.txt

## 数据格式

预训练数据应为JSONL格式，每行包含一个JSON对象：

```json
{"text": "这是一段用于预训练的纯文本内容..."}
```

## 输出文件

训练完成后，会在指定目录生成以下文件：

- `pretrain_512.pth` (14G模式) 或 `pretrain_768.pth` (24G模式) - 模型权重文件
- 训练日志和wandb记录（如果启用）

## 故障排除

### 常见问题

1. **显存不足错误**
   - 尝试降低batch size或序列长度
   - 使用梯度累积来模拟更大的batch size

2. **数据文件不存在**
   - 检查数据文件路径是否正确
   - 使用 `-d` 参数指定正确的路径

3. **CUDA不可用**
   - 检查CUDA驱动和PyTorch安装
   - 脚本会自动降级到CPU模式

### 性能优化建议

- 使用NVMe SSD存储数据文件以获得更快的读取速度
- 启用混合精度训练（脚本已内置）
- 使用wandb监控训练过程
- 定期保存检查点以防训练中断

## 扩展自定义

如需自定义模型参数，可以直接修改脚本中的配置部分：

```bash
# 在脚本中修改这些参数
HIDDEN_SIZE=512      # 隐藏层维度
NUM_LAYERS=8         # Transformer层数
MAX_SEQ_LEN=512      # 最大序列长度
BATCH_SIZE=16        # Batch Size
ACCUMULATION_STEPS=4 # 梯度累积步数
```

## 相关脚本

- `train_pretrain.py`: 核心训练脚本
- `model/model.py`: 模型定义
- `dataset/lm_dataset.py`: 数据集处理

## 技术支持

如有问题，请参考项目文档或联系开发团队。