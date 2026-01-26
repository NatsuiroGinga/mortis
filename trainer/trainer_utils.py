"""
MiniMind 训练工具函数集合

本模块提供训练过程中常用的工具函数：

【核心功能】
┌──────────────────────────────────────────────────────────────────┐
│  init_distributed_mode()  - 初始化分布式训练环境 (DDP)            │
│  setup_seed()             - 设置随机种子，确保可复现性            │
│  init_model()             - 初始化模型和分词器                    │
│  lm_checkpoint()          - 保存/加载检查点 (支持断点续训)        │
│  get_lr()                 - 余弦退火学习率调度                    │
│  SkipBatchSampler         - 支持跳过已训练 batch 的采样器         │
└──────────────────────────────────────────────────────────────────┘

【分布式训练流程】
1. init_distributed_mode() 初始化 NCCL 后端
2. setup_seed(42 + rank) 设置不同的随机种子
3. DistributedSampler 分配数据到各 GPU
4. DistributedDataParallel 包装模型
5. lm_checkpoint() 只在主进程保存检查点
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_mortis import MortisForCausalLM


def get_model_params(model, config):
    """
    计算并打印模型参数量。

    【MoE 模型参数计算】
    对于 MoE 模型，需要区分总参数量和激活参数量：
    - 总参数 = 基础参数 + (专家参数 × 专家数量) + (共享专家参数 × 共享专家数)
    - 激活参数 = 基础参数 + (专家参数 × 每token激活专家数) + (共享专家参数 × 共享专家数)

    【示例】
    MiniMind2-MoE: 145M-A45M (总参数145M，每次激活45M)

    Args:
        model: 模型实例
        config: 模型配置
    """
    # 总参数量 (百万)
    total = sum(p.numel() for p in model.parameters()) / 1e6

    # 获取 MoE 配置
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))  # 路由专家数
    n_active = getattr(config, 'num_experts_per_tok', 0)  # 每 token 激活的专家数
    n_shared = getattr(config, 'n_shared_experts', 0)  # 共享专家数

    # 计算单个专家的参数量
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6

    # 计算基础参数 (非专家部分)
    base = total - (expert * n_routed) - (shared_expert * n_shared)

    # 计算激活参数量
    active = base + (expert * n_active) + (shared_expert * n_shared)

    # 打印参数量
    if active < total:
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')  # MoE 模型
    else:
        Logger(f'Model Params: {total:.2f}M')  # Dense 模型


def is_main_process():
    """
    判断当前进程是否为主进程。

    【用途】
    - 日志打印：只在主进程打印，避免重复输出
    - 模型保存：只在主进程保存，避免多进程同时写入
    - wandb 记录：只在主进程记录

    Returns:
        bool: True 表示主进程 (rank=0 或非分布式模式)
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """
    主进程日志打印函数。

    Args:
        content: 要打印的内容
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    余弦退火学习率调度 (Cosine Annealing with Warmup)。

    【公式】
    lr(t) = lr_base × (0.1 + 0.45 × (1 + cos(π × t / T)))

    【学习率变化】
    - t=0:      lr × 1.0   (起始)
    - t=T/2:    lr × 0.55  (中点)
    - t=T:      lr × 0.1   (终点)

    【图示】
    lr  ┃ ▂▂▂
        ┃▂▂▂   ▂▂
        ┃         ▂▂
        ┃            ▂▂▂▂▂
        ┗━━━━━━━━━━━━━━━━━━━ step
         0               T

    Args:
        current_step: 当前步数
        total_steps: 总步数
        lr: 基础学习率

    Returns:
        float: 当前步的学习率
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """
    初始化分布式训练环境 (DDP)。

    【工作原理】
    1. 检查环境变量 RANK，判断是否为分布式模式
    2. 如果是分布式模式，初始化 NCCL 进程组
    3. 设置当前进程使用的 GPU 设备

    【环境变量】
    - RANK: 全局进程编号 (0, 1, 2, ...)
    - LOCAL_RANK: 本机进程编号 (通常等于使用的 GPU 编号)
    - WORLD_SIZE: 总进程数

    【启动方式】
    torchrun --nproc_per_node=4 train_pretrain.py

    Returns:
        int: local_rank (非 DDP 模式返回 0)
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式

    # 初始化 NCCL 后端 (NVIDIA GPU 通信库)
    dist.init_process_group(backend="nccl")

    # 获取本地 rank 并设置对应 GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def setup_seed(seed: int):
    """
    设置所有随机种子，确保实验可复现。

    【设置的随机源】
    - Python random 模块
    - NumPy 随机数生成器
    - PyTorch CPU 随机数生成器
    - PyTorch CUDA 随机数生成器 (所有 GPU)
    - cuDNN 确定性模式

    【注意】
    分布式训练时，每个进程应使用不同的种子：
    setup_seed(42 + rank)

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
    torch.backends.cudnn.deterministic = True  # 确定性模式
    torch.backends.cudnn.benchmark = False  # 关闭自动优化


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    保存或加载训练检查点，支持断点续训。

    【保存模式】(model is not None)
    保存两个文件：
    1. {weight}_{hidden_size}.pth          - 纯模型权重 (用于推理)
    2. {weight}_{hidden_size}_resume.pth   - 完整训练状态 (用于续训)

    【加载模式】(model is None)
    从 resume 文件加载训练状态

    【断点续训状态】
    - model: 模型权重
    - optimizer: 优化器状态
    - epoch: 当前 epoch
    - step: 当前 step
    - world_size: GPU 数量 (用于调整 step)
    - wandb_id: wandb 运行 ID (用于续训 wandb)
    - **kwargs: 其他需要保存的对象 (如 scaler, scheduler)

    【GPU 数量变化处理】
    如果续训时 GPU 数量变化，自动调整 step：
    new_step = old_step × old_world_size / new_world_size

    Args:
        lm_config: 模型配置
        weight: 权重名称前缀
        model: 模型实例 (None 表示加载模式)
        optimizer: 优化器
        epoch: 当前 epoch
        step: 当前 step
        wandb: wandb 实例
        save_dir: 保存目录
        **kwargs: 其他需要保存的对象

    Returns:
        dict or None: 加载模式下返回检查点数据，否则返回 None
    """
    os.makedirs(save_dir, exist_ok=True)

    # 构建文件路径
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # ========== 保存模式 ==========
        from torch.nn.parallel import DistributedDataParallel

        # 获取模型状态字典 (处理 DDP 包装)
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()

        # 转换为 half 精度并移到 CPU (节省空间)
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 使用临时文件 + rename 确保原子性写入
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 获取 wandb run ID
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 构建完整的续训数据
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }

        # 保存额外的对象 (如 scaler, scheduler)
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        # 原子性保存续训文件
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 清理内存
        del state_dict, resume_data
        torch.cuda.empty_cache()

    else:
        # ========== 加载模式 ==========
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')

            # 处理 GPU 数量变化
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                # 调整 step: 保持总处理样本数不变
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')

            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='./model', save_dir='../out', device='cuda'):
    """
    初始化模型和分词器。

    【加载流程】
    1. 加载分词器
    2. 创建模型实例
    3. 加载预训练权重 (如果指定)
    4. 打印参数量信息
    5. 移动到指定设备

    Args:
        lm_config: 模型配置
        from_weight: 预训练权重名称 ('none' 表示从头训练)
        tokenizer_path: 分词器路径
        save_dir: 权重保存目录
        device: 训练设备

    Returns:
        tuple: (model, tokenizer)
    """
    # 加载分词器（转换为绝对路径以避免HuggingFace验证错误）
    tokenizer_abs_path = os.path.abspath(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_abs_path, local_files_only=True, trust_remote_code=True)

    # 创建模型
    model = MortisForCausalLM(lm_config)

    # 加载预训练权重
    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)  # strict=False 允许部分加载

    # 打印参数量
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    支持跳过已训练 batch 的采样器，用于断点续训。

    【工作原理】
    包装另一个 sampler，跳过前 skip_batches 个 batch。

    【使用场景】
    断点续训时，跳过已训练的 batch：
    - 上次训练到 step=1000
    - 续训时设置 skip_batches=1000
    - 直接从 step=1001 开始训练

    【示例】
    sampler = DistributedSampler(dataset)
    batch_sampler = SkipBatchSampler(sampler, batch_size=32, skip_batches=1000)
    loader = DataLoader(dataset, batch_sampler=batch_sampler)

    Args:
        sampler: 底层采样器 (如 DistributedSampler)
        batch_size: batch 大小
        skip_batches: 要跳过的 batch 数量
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    # 跳过前 skip_batches 个 batch
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # 处理最后一个不完整的 batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
