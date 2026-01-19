# Mortis

A lightweight language model training framework - simplified version of MiniMind for educational purposes and rapid prototyping.

> **Mortis** (拉丁语"死亡") 象征着剥繁就简，只保留核心精髓，让代码更"纯粹"。

---

## 概述

Mortis 是 MiniMind 的极简实现，保留了完整的语言模型训练能力，但减少了代码复杂度和依赖项。适合用于：

- **学习 LLM 训练原理**：代码结构清晰，易于理解
- **快速实验原型**：小而美，可快速验证想法
- **教学演示**：核心组件约 700 行，便于讲解

### 与 MiniMind 的对比

| 特性 | Mortis | MiniMind |
|------|--------|----------|
| 代码量 | ~700 行 | ~5000+ 行 |
| 模型架构 | Dense | Dense + MoE |
| 训练方式 | 预训练 | 预训练 + SFT + RLHF + LoRA |
| 训练脚本 | 1 个 | 7+ 个 |
| 适用场景 | 学习/原型 | 生产/研究 |

---

## 快速开始

### 环境准备

```bash
# 1. 克隆项目（包含 mortis 和 minimind）
git clone https://github.com/NatsuiroGinga/mortis.git
cd mortis

# 2. 安装依赖
cd ..
pip install -r minimind/requirements.txt
cd mortis

# 3. 准备数据
# 确保数据文件位于 ./dataset/pretrain_hq.jsonl
```

### 一键启动训练

```bash
# 14G 显存模式
chmod +x scripts/pretrain.sh
./scripts/pretrain.sh 14g

# 24G 显存模式
./scripts/pretrain.sh 24g
```

### 手动训练

```bash
python trainer/train_pretrain.py \
    --data_path ./dataset/pretrain_hq.jsonl \
    --epochs 1 \
    --batch_size 32 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --max_seq_len 340
```

---

## 项目结构

```
mortis/
├── model/                       # 模型实现
│   └── model.py                # 核心模型定义（~500 行）
│       ├── MortisConfig       # 模型配置
│       ├── RMSNorm            # 层归一化
│       ├── precompute_freqs_cis()  # RoPE + YaRN
│       ├── Attention          # 注意力机制 (GQA)
│       ├── FeedForward        # SwiGLU 前馈网络
│       ├── TransformerBlock   # Transformer 块
│       ├── MortisModel        # 语言模型
│       └── MortisForCausalLM  # 因果语言模型
│
├── dataset/                     # 数据处理
│   └── lm_dataset.py          # 预训练数据集
│       └── PretrainDataset    # 纯文本预训练
│
├── trainer/                     # 训练相关
│   ├── train_pretrain.py      # 预训练脚本
│   └── trainer_utils.py       # 训练工具函数
│
├── scripts/                     # 辅助脚本
│   ├── pretrain.sh            # 一键训练脚本
│   └── README.md              # 脚本使用说明
│
├── out/                         # 模型输出（.gitignore）
├── checkpoints/                 # 训练检查点（.gitignore）
├── .gitignore                   # Git 忽略规则
└── README.md                    # 本文档
```

---

## 模型参数配置

### Mortis 支持的配置

| 配置 | hidden_size | num_layers | 参数量 | 显存需求 |
|------|-------------|-----------|--------|---------|
| Mortis-Small | 512 | 8 | ~26M | ~8GB |
| Mortis-Medium | 768 | 12 | ~78M | ~14GB |
| Mortis-Base | 768 | 16 | ~104M | ~24GB |

### 核心超参数

```python
MortisConfig(
    hidden_size=512,              # 隐藏层维度
    num_hidden_layers=8,          # Transformer 层数
    num_attention_heads=8,        # 注意力头数
    num_key_value_heads=2,        # KV 头数 (GQA)
    vocab_size=6400,              # 词汇表大小
    max_position_embeddings=32768 # 最大序列长度
)
```

---

## 核心组件

### 1. 模型架构

```
MortisForCausalLM
├── MortisModel
│   ├── Embedding (vocab_size=6400)
│   ├── TransformerBlock × N
│   │   ├── RMSNorm (Pre-Norm)
│   │   ├── Attention (GQA + RoPE)
│   │   └── FeedForward (SwiGLU)
│   └── RMSNorm (Final)
└── LM Head
```

### 2. 关键技术

- **GQA (Grouped Query Attention)**: 分组查询注意力，减少显存
- **RoPE + YaRN**: 旋转位置编码 + 长上下文外推
- **SwiGLU**: 交换门控线性单元激活函数
- **Pre-Norm**: 前置层归一化，稳定训练

### 3. 训练特性

- 混合精度训练 (bfloat16/float16)
- 梯度累积
- 梯度裁剪
- 余弦退火学习率调度
- DDP 多 GPU 训练支持
- 断点续训

---

## 详细文档

- **训练脚本说明**: [scripts/README.md](scripts/README.md)
- **完整项目文档**: [../CLAUDE.md](../CLAUDE.md)

---

## 使用示例

### 预训练

```bash
# 使用脚本（推荐）
./scripts/pretrain.sh 14g -e 2 -w

# 直接运行 Python
python trainer/train_pretrain.py \
    --data_path ./dataset/pretrain_hq.jsonl \
    --epochs 2 \
    --batch_size 40 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --max_seq_len 512 \
    --use_wandb
```

### 断点续训

```bash
python trainer/train_pretrain.py \
    --from_resume 1 \
    --epochs 5
```

### 多 GPU 训练

```bash
torchrun --nproc_per_node=4 trainer/train_pretrain.py
```

---

## 数据格式

预训练数据为 JSONL 格式：

```json
{"text": "这是一段训练文本..."}
```

示例数据文件：`./dataset/pretrain_hq.jsonl`

---

## 输出文件

训练完成后生成：

```
out/
└── pretrain_768.pth          # 模型权重

checkpoints/
├── pretrain_768.pth          # 推理权重
├── pretrain_768_resume.pth   # 完整训练状态（用于续训）
└── ...
```

---

## 技术支持

### 常见问题

1. **显存不足**
   - 降低 `batch_size` 或 `max_seq_len`
   - 使用梯度累积保持有效 batch size

2. **分词器找不到**
   - 确保 `../minimind/model` 目录存在
   - 检查 Transformer 版本兼容性

3. **训练速度慢**
   - 使用 bfloat16 混合精度
   - 启用数据加载多进程
   - 使用 NVMe SSD 存储数据

### 获取帮助

- 查看 [scripts/README.md](scripts/README.md)
- 参考 [Minimind 原项目](https://github.com/jianzhnie/MiniMind)
- 检查训练日志输出

---

## 许可证

本项目基于 [MiniMind](https://github.com/jianzhnie/MiniMind) 项目，简化版本用于学习和研究目的。

---

## 致谢

- [MiniMind](https://github.com/jianzhnie/MiniMind) - 原始项目
- [Llama](https://github.com/facebookresearch/llama) - 架构参考
- PyTorch & Transformers - 核心框架
