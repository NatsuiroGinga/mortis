# Mortis 预训练脚本使用说明

## 概述

`pretrain.sh` 是一个智能的预训练启动脚本，支持不同显存容量的 GPU 配置。脚本会自动根据选择的模式调整模型参数和训练配置，以最大化利用可用显存。

## 快速开始

### 基本用法

```bash
# 14G显存模式 (适合NVIDIA L20/4060 Ti 16G等显卡)
./scripts/pretrain.sh 14g

# 24G显存模式 (适合RTX 3090/4090等显卡)
./scripts/pretrain.sh 24g
```

### 高级用法

```bash
# 14G模式，训练2轮，启用wandb日志
./scripts/pretrain.sh 14g -e 2 -w

# 24G模式，自定义数据路径和保存目录
./scripts/pretrain.sh 24g -d ./dataset/pretrain_hq.jsonl -s ./out

# 自定义学习率
./scripts/pretrain.sh 14g -l 1e-3 -e 3
```

## 模式配置详情

### 14G显存模式

- **适用显卡**: NVIDIA L20, RTX 4060 Ti 16G, RTX 3070 16G等
- **模型配置**:
  - 隐藏层维度: 768
  - Transformer层数: 12
  - 最大序列长度: 512
  - 模型参数量: ~78M
- **训练配置**:
  - Batch Size: 40
  - 梯度累积步数: 1
  - 有效Batch Size: 40

### 24G显存模式

- **适用显卡**: RTX 3090 24G, RTX 4090 24G, A5000 24G等
- **模型配置**:
  - 隐藏层维度: 768
  - Transformer层数: 16
  - 最大序列长度: 1024
  - 模型参数量: ~104M
- **训练配置**:
  - Batch Size: 32
  - 梯度累积步数: 2
  - 有效Batch Size: 64

## 参数说明

### 必需参数
- `模式` (14g/24g): 选择显存模式

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-d, --data-path` | 数据文件路径 | `./dataset/pretrain_hq.jsonl` |
| `-e, --epochs` | 训练轮数 | `1` |
| `-l, --learning-rate` | 学习率 | `5e-4` |
| `-s, --save-dir` | 模型保存目录 | `../out` |
| `-w, --wandb` | 启用wandb日志记录 | 禁用 |
| `-h, --help` | 显示帮助信息 | - |

## 环境要求

### 硬件要求
- **GPU**: NVIDIA显卡，支持CUDA 11.7+
- **显存**: 至少14GB可用显存
- **内存**: 建议32GB以上
- **存储**: 建议100GB以上SSD空间

### 软件要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Python | 3.8 | 3.10+ |
| PyTorch | 2.0 | 2.1+ |
| CUDA | 11.7 | 12.1+ |
| Transformers | 4.30+ | - |

### 依赖安装

```bash
# 克隆 minimind 项目获取依赖和 tokenizer
cd ..
pip install -r minimind/requirements.txt

# 或手动安装核心依赖
pip install torch transformers accelerate
```

## 数据格式

预训练数据应为 JSONL 格式，每行包含一个 JSON 对象：

```json
{"text": "这是一段用于预训练的纯文本内容..."}
```

数据文件默认位置: `./dataset/pretrain_hq.jsonl`

## 输出文件

训练完成后，会在指定目录生成以下文件：

```
out/
├── pretrain_768.pth              # 模型权重文件 (用于推理)
└── checkpoints/                  # 训练检查点目录 (用于断点续训)
    ├── pretrain_768_resume.pth   # 完整训练状态
    └── ...
```

## 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 脚本检查环境（GPU、数据文件、Python版本）               │
│  2. 根据模式自动配置训练参数                                 │
│  3. 确认配置后开始训练                                       │
│  4. 定期保存检查点（每500步）                                │
│  5. 训练完成后显示模型信息                                   │
└─────────────────────────────────────────────────────────────┘
```

## 断点续训

训练中断后，使用以下命令继续：

```bash
# 修改 trainer/train_pretrain.py，设置 from_resume=1
python trainer/train_pretrain.py --from_resume 1 ...

# 或直接修改 train_pretrain.py 的默认参数
```

## 故障排除

### 常见问题

1. **显存不足错误**
   ```bash
   # 检查显存使用情况
   nvidia-smi

   # 尝试降低batch size（修改脚本中的BATCH_SIZE）
   # 或使用梯度累积来保持有效batch size
   ```

2. **数据文件不存在**
   ```bash
   # 检查数据文件路径
   ls -la ./dataset/

   # 使用 -d 参数指定正确的路径
   ./scripts/pretrain.sh 14g -d ./dataset/pretrain_hq.jsonl
   ```

3. **CUDA不可用**
   ```bash
   # 检查驱动和CUDA
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"

   # 脚本会自动降级到CPU模式（速度较慢）
   ```

4. **分词器找不到**
   ```bash
   # 确保可以访问 minimind/model 目录
   # trainer/train_pretrain.py 默认从 ../minimind/model 加载
   ```

### 性能优化建议

- 使用 NVMe SSD 存储数据文件以获得更快的读取速度
- 启用混合精度训练（脚本已默认使用 bfloat16）
- 使用 wandb 监控训练过程（`-w` 参数）
- 定期保存检查点以防训练中断
- 多 GPU 训练使用 DDP: `torchrun --nproc_per_node=4 trainer/train_pretrain.py`

## 扩展自定义

如需自定义模型参数，可以直接修改 `scripts/pretrain.sh` 中的配置部分：

```bash
# 在 case 部分修改这些参数
HIDDEN_SIZE=768      # 隐藏层维度 (512/640/768)
NUM_LAYERS=12        # Transformer层数 (8/12/16)
MAX_SEQ_LEN=512      # 最大序列长度
BATCH_SIZE=40        # Batch Size
ACCUMULATION_STEPS=1 # 梯度累积步数
```

## 项目结构

```
mortis/
├── model/               # 模型定义
│   └── model.py        # MortisConfig, MortisForCausalLM 等
├── dataset/             # 数据集
│   └── pretrain_hq.jsonl
├── trainer/             # 训练脚本
│   ├── train_pretrain.py
│   └── trainer_utils.py
├── scripts/             # 辅助脚本
│   ├── pretrain.sh     # 预训练启动脚本
│   └── README.md       # 本文档
├── out/                 # 模型输出目录
└── checkpoints/         # 训练检查点
```

## 相关文档

- [Mortis 项目 README](../README.md)
- [MiniMind 主项目文档](../CLAUDE.md)
- [训练脚本源码](../trainer/train_pretrain.py)

## 技术特性

- **GQA (Grouped Query Attention)**: 减少注意力计算的显存占用
- **RoPE + YaRN**: 支持长上下文外推
- **SwiGLU**: 更高效的激活函数
- **混合精度训练**: bfloat16/float16 加速
- **DDP 支持**: 多 GPU 分布式训练
- **断点续训**: 完整保存训练状态

## 许可证与致谢

本项目基于 [MiniMind](https://github.com/jianzhnie/MiniMind) 项目，简化版本用于学习目的。
