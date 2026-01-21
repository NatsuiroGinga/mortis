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

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}", add_special_tokens=False).input_ids
    
    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, cs):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                tools=tools)

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample["conversations"])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self.generate_loss_mask(input_ids)
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask














