"""
Mortis 数据集模块

本模块提供预训练和微调所需的数据集类。

【训练流程】
┌──────────────┐    ┌──────────────┐
│ PretrainData │ -> │   SFTDataset │
│   (预训练)    │    │  (监督微调)   │
└──────────────┘    └──────────────┘

【数据格式】
1. PretrainDataset: {"text": "纯文本内容..."}
2. SFTDataset: {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

【Loss Mask 机制】
- 预训练: 所有非 padding 位置都计算损失
- SFT: 只在 assistant 回复部分计算损失
"""

from torch.utils.data import Dataset
import torch
import os
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset

# 禁用 tokenizers 并行，避免多进程死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """
    预训练数据集，用于从头训练语言模型。

    【数据格式】
    输入 JSONL 文件，每行一个 JSON 对象：
    {"text": "这是一段用于预训练的纯文本..."}

    【处理流程】
    原始文本 → Tokenize → Padding → 构建 (X, Y, loss_mask)

    【示例】
    输入: "今天天气很好"
    Tokenized: [101, 234, 567, 890, 102, 0, 0, 0]  (padding 到 max_length)
    X:         [101, 234, 567, 890, 102, 0, 0]     (去掉最后一个)
    Y:         [234, 567, 890, 102, 0, 0, 0]       (去掉第一个)
    loss_mask: [1,   1,   1,   1,   0, 0, 0]       (padding 位置为 0)

    【参数】
    - data_path: JSONL 文件路径
    - tokenizer: 分词器
    - max_length: 最大序列长度 (默认 512)
    """
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 库加载 JSONL 文件
        self.samples = load_dataset("json",
                                    data_files=data_path,
                                    split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # ========== 1. Tokenize 原始文本 ==========
        # padding='max_length': 填充到固定长度
        # truncation=True: 超长截断
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding.input_ids.squeeze()  # [max_length]

        # ========== 2. 构建 loss_mask ==========
        # 只在非 padding 位置计算损失
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # ========== 3. 构建训练数据 ==========
        # 语言模型任务: 用前 n-1 个 token 预测后 n-1 个 token
        # X: input_ids[:-1]  (输入)
        # Y: input_ids[1:]   (目标)
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐到 Y 的位置

        return X, Y, loss_mask
