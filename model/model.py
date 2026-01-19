"""
MiniMind 是一个从零训练的超小型语言模型（25.8M-145M 参数），采用现代 Transformer 架构。

【整体架构】
┌─────────────────────────────────────────────────────────────┐
│                   MiniMindForCausalLM                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    MiniMindModel                      │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │              Embedding Layer                    │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │                        ↓                              │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │         MiniMindBlock × N (8/16 层)             │  │  │
│  │  │  ┌───────────────────────────────────────────┐  │  │  │
│  │  │  │ RMSNorm → Attention (GQA + RoPE + Flash)  │  │  │  │
│  │  │  │              ↓ + residual                 │  │  │  │
│  │  │  │ RMSNorm → FFN (SwiGLU) / MoE              │  │  │  │
│  │  │  │              ↓ + residual                 │  │  │  │
│  │  │  └───────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │                        ↓                              │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │              Final RMSNorm                      │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │          LM Head (权重与 Embedding 共享)               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

【核心特性】
- RMSNorm：比 LayerNorm 计算更快，效果相当
- RoPE + YaRN：旋转位置编码 + 长度外推（支持 32K 上下文）
- GQA (Grouped Query Attention)：减少 KV 缓存内存
- Flash Attention：内存高效的注意力机制
- SwiGLU：优于 GELU 的激活函数
- MoE (Mixture of Experts)：稀疏激活，参数效率高
- 权重共享：Embedding 和 LM Head 共享权重，减少 ~30% 参数
"""

import math

import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


class MortisConfig(PretrainedConfig):
    """
    MiniMind 模型配置类，继承 HuggingFace PretrainedConfig，完全兼容 HuggingFace 生态。

    【模型规格】
    - MiniMind2-Small:  hidden_size=512, num_layers=8,  ~26M 参数
    - MiniMind2:         hidden_size=768, num_layers=16, ~104M 参数
    - MiniMind2-MoE:      hidden_size=640, num_layers=8,  ~145M 参数（MoE架构）
    """
    model_type = "minimind"
    def __init__(
        self,
        drop_out: float = 0.0, # Dropout 概率，用于防止过拟合（训练时随机丢弃部分神经元）
        bos_token_id: int = 1, # Beginning of Sequence Token ID，句子开始标记的索引
        eos_token_id: int = 2, # End of Sequence Token ID，句子结束标记的索引
        hidden_act: str = "silu", # 隐藏层激活函数, 默认使用 SILU(Swish)激活函数
        hidden_size: int = 512, # 隐藏层维度, 控制模型宽度和参数量(Small: 512, Base: 768)
        intermediate_size: int | None = None, # FFN中间层维度, 默认为hidden_size * 8 / 3 并对齐到64倍数
        max_position_embeddings: int = 32768, # 最大序列长度, 支持的最大上下文长度(通过YaRN外推支持32K)
        num_attention_heads: int = 8, # 注意力头数, 多头自注意力的头数(Small: 8, Base: 12)
        num_hidden_layers: int = 8, # Transformer层数, 控制模型深度(Small: 8, Base: 16)
        num_key_value_heads: int = 2, # KV头数, GQA(分组查询注意力)中的键值头数(通常为num_attention_heads/4)
        vocab_size: int = 6400, # 词表大小, tokenizer的词汇表大小(影响embedding和LM head参数量)
        rms_norm_eps: float = 1e-05, # RMSNorm的epsilon值, 防止除零错误
        rope_theta: int = 100_0000.0, # RoPE 的基础频率，控制位置编码的衰减速度（影响长文本建模能力）
        inference_rope_scaling: bool = False, # 推理时是否启用 RoPE 长度外推，用于处理超出训练长度的序列
        flash_attn: bool = True, # 是否使用 Flash Attention，提高注意力计算效率
        # MOE配置参数
        # 当 use_moe = False 时，以下参数无效
        use_moe: bool = False, # 是否使用 MoE 架构
        num_experts_per_tok: int = 2, # 每个 token 选择的专家数量, 稀疏激活参数
        n_routed_experts: int = 4, # 总路由专家数量, MOE中的专家总数
        n_shared_experts: int = 1, # 共享专家数量, 所有token都会经过的共享专家
        scoring_func: str = "softmax", # 门控评分函数, 用于选择专家的评分机制
        aux_loss_alpha: float = 0.001, # 辅助损失权重, 负载均衡损失的权重 (防止所有token涌向同一专家)
        seq_aux: bool = True, # 是否在序列级别计算辅助损失, True为序列级, False为全局级
        norm_topk_prob: bool = True, # 是否标准化 top-k 概率, 确保选中的专家权重和为1\
        **kwargs
    ):
        super().__init__(**kwargs)
        self.drop_out = drop_out
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 16 * 2048 = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        # MOE配置参数
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (均方根归一化)

    [为什么使用RMSNorm而不是LayerNorm?]
    1. 计算更快, 省去了均值计算, 只需计算均方根
    2. 效果相当: 在LLM中实验证明效果与LayerNorm相当
    3. 数值稳定: x.float() 确保在低精度训练时的稳定性

    [数学公式]
    RMSNorm(x) = (x / √(1/n * Σ(x_i²) + ε)) * γ
    其中γ是可学习的缩放参数 (self.weight), ε 是极小值防止除零
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # x shape: (batch_size, seq_len, dim)
        # x.pow(2).mean(-1, keepdim=True) shape: (batch_size, seq_len, 1)
        # 计算均方根归一化: x / √(1/n * Σ(x_i²) + ε)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # 使用 flaot32 计算确保数值稳定性, 然后转回原始类型
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim: int,
                         end: int = int(32 * 1024),
                         rope_base: float = 1e6,
                         rope_scaling: dict | None = None) -> torch.Tensor:
    """
    预计算 RoPE 旋转位置编码的复数频率表。

    Args:
        dim: 注意力头的维度 (head_dim)
        end: 最大序列长度
        rope_base: RoPE 基础频率
        rope_scaling: YaRN 长度外推配置，None 表示使用标准 RoPE

    Returns:
        freq_cis: 复数频率表 [end, dim // 2]，类型为 complex64
    """
    # 计算基础频率: θ_i = 1 / (base^(2i/dim))
    # freqs shape: [dim // 2]
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))

    # 标准 RoPE（无 YaRN 外推）
    if rope_scaling is None:
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()  # [end, dim // 2]
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    # YaRN 长度外推 (Yet another RoPE extension)
    # 通过分段线性插值实现长度外推，从 orig_max 扩展到 orig_max * factor
    # 参考论文: https://arxiv.org/abs/2309.00071
    orig_max, factor, beta_fast, beta_slow = (
        rope_scaling.get("original_max_position_embeddings", 2048),
        rope_scaling.get("factor", 16),
        rope_scaling.get("beta_fast", 32),
        rope_scaling.get("beta_slow", 1),
    )

    # 计算维度阈值的反函数: r(d) = (d * log(L / (2π * b))) / (2 * log(base))
    # 用于确定哪些维度需要插值
    # - low: beta_fast 对应的维度阈值（高频边界）
    # - high: beta_slow 对应的维度阈值（低频边界）
    inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
    low = max(math.floor(inv_dim(beta_fast)), 0)
    high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

    # 计算线性斜坡 γ (ramp):
    # - 维度 < low: γ = 0，保持原频率不变
    # - low <= 维度 <= high: γ 从 0 线性增加到 1
    # - 维度 > high: γ = 1，完全缩放
    ramp = torch.clamp(
        (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
        0, 1
    )

    # 应用 YaRN 缩放: f'(i) = f(i) * ((1-γ) + γ/factor)
    # - γ = 0 时: f'(i) = f(i)，保持不变
    # - γ = 1 时: f'(i) = f(i) / factor，完全缩放
    # - 0 < γ < 1: 平滑插值
    freqs = freqs * (1 - ramp + ramp / factor)

    # 构建位置-频率矩阵并转为复数
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [end, dim // 2]
    freq_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freq_cis

def reshape_for_broadcast(freq_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # x: [bs, seq_len, num_heads, head_dim // 2]
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freq_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freq_cis.view(*shape) # (1, T, 1, head_dim // 2)

def apply_rotary_emb(xq: torch.Tensor, 
                     xk: torch.Tensor, 
                     freq_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freq_cis = reshape_for_broadcast(freq_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freq_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freq_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    """
    多头注意力机制，支持 GQA (Grouped Query Attention) 和 Flash Attention。

    【GQA (Grouped Query Attention)】
    - 标准 MHA: num_q_heads = num_kv_heads (每个 Q 头对应一个 KV 头)
    - GQA: num_q_heads > num_kv_heads (多个 Q 头共享一组 KV 头)
    - 优势: 减少 KV Cache 内存占用，加速推理，效果接近 MHA

    【计算流程】
    1. 线性投影: x → Q, K, V
    2. 分头: reshape 为 [batch, seq, heads, head_dim]
    3. RoPE: 对 Q, K 应用旋转位置编码
    4. KV Cache: 拼接历史 KV (推理时)
    5. GQA 扩展: 将 KV 头复制 n_rep 次以匹配 Q 头数
    6. 注意力计算: Flash Attention 或手动实现
    7. 输出投影: concat 后通过 Wo 投影回 hidden_size
    """
    def __init__(self, args: MortisConfig):
        super().__init__()
        # GQA 配置: num_key_value_heads < num_attention_heads
        # 如果 num_key_value_heads 为 None，则退化为标准 MHA
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads  # Q 头数
        self.num_key_value_heads = args.num_key_value_heads  # KV 头数
        self.n_rep = self.n_local_heads // self.num_key_value_heads  # 每个 KV 头被多少个 Q 头共享
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度

        # Q 投影: hidden_size → num_q_heads * head_dim
        self.wq = nn.Linear(args.hidden_size,
                            args.num_attention_heads * self.head_dim,
                            bias = False)
        # K 投影: hidden_size → num_kv_heads * head_dim (GQA 时比 Q 小)
        self.wk = nn.Linear(args.hidden_size,
                            self.num_key_value_heads * self.head_dim,
                            bias = False)
        # V 投影: hidden_size → num_kv_heads * head_dim (GQA 时比 Q 小)
        self.wv = nn.Linear(args.hidden_size,
                            self.num_key_value_heads * self.head_dim,
                            bias = False)
        # 输出投影: num_q_heads * head_dim → hidden_size
        self.wo = nn.Linear(args.num_attention_heads * self.head_dim,
                            args.hidden_size,
                            bias = False)

        self.attn_dropout = nn.Dropout(args.drop_out)  # 注意力权重 dropout
        self.resid_dropout = nn.Dropout(args.drop_out)  # 残差连接前的 dropout
        self.drop_out = args.drop_out

        # 检测是否支持 Flash Attention (PyTorch 2.0+)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                freq_cis: torch.Tensor,
                past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache: bool = False,
                attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            freq_cis: RoPE 频率表 [seq_len, head_dim // 2] (复数)
            past_key_value: KV Cache (k, v)，用于推理加速
            use_cache: 是否返回更新后的 KV Cache
            attention_mask: padding mask [batch_size, seq_len]，1=有效，0=padding

        Returns:
            output: 注意力输出 [batch_size, seq_len, hidden_size]
            past_kv: 更新后的 KV Cache (如果 use_cache=True)
        """
        batch_size, seq_len, _ = x.shape

        # ========== 1. 线性投影 ==========
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # ========== 2. 分头 reshape ==========
        # Q: [bs, seq, hidden] → [bs, seq, n_q_heads, head_dim]
        # K/V: [bs, seq, hidden] → [bs, seq, n_kv_heads, head_dim]
        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # ========== 3. RoPE 旋转位置编码 ==========
        # 只对 Q 和 K 应用 RoPE，V 不需要位置信息
        xq, xk = apply_rotary_emb(xq, xk, freq_cis)

        # ========== 4. KV Cache 处理 (推理时) ==========
        # 将当前的 K, V 与历史的 K, V 拼接
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # ========== 5. 转置并扩展 KV 头 (GQA) ==========
        # transpose: [bs, seq, heads, dim] → [bs, heads, seq, dim]
        xq = xq.transpose(1, 2)
        # repeat_kv: 将 KV 头复制 n_rep 次以匹配 Q 头数
        # [bs, seq, n_kv_heads, dim] → [bs, seq, n_q_heads, dim] → [bs, n_q_heads, seq, dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # ========== 6. 注意力计算 ==========
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention: O(N) 内存，更快
            # 条件: seq_len > 1 且没有 padding (因为 is_causal=True 不支持自定义 mask)
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.drop_out if self.training else 0.0,
                is_causal=True  # 自动应用 causal mask
            )
        else:
            # 手动实现注意力 (支持自定义 mask)
            # scores = Q @ K^T / sqrt(head_dim)
            scores = xq @ xk.transpose(2, 3) / math.sqrt(self.head_dim)

            # Causal Mask: 上三角设为 -inf，防止看到未来 token
            # [[0, -inf, -inf],
            #  [0,   0, -inf],
            #  [0,   0,   0]]
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq] 用于广播
            scores = scores + causal_mask

            # Padding Mask: 忽略 padding token
            # attention_mask: [bs, seq], 1=有效, 0=padding
            # 转换: 1→0 (不变), 0→-1e9 (屏蔽)
            if attention_mask is not None:
                # [bs, seq] → [bs, 1, 1, seq] 用于广播到 [bs, heads, seq, seq]
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 1 → 0, 0 → -1e9
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # Softmax + Dropout
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)

            # 加权求和: output = softmax(scores) @ V
            output = scores @ xv

        # ========== 7. 合并多头并输出投影 ==========
        # [bs, heads, seq, dim] → [bs, seq, heads, dim] → [bs, seq, hidden]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.resid_dropout(self.wo(output))

        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, args: MortisConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.w_up = nn.Linear(args.hidden_size,
                              args.intermediate_size,
                              bias=False)
        self.w_down = nn.Linear(args.intermediate_size,
                                args.hidden_size,
                                bias=False)
        self.w_gate = nn.Linear(args.hidden_size,
                                args.intermediate_size,
                                bias=False)
        self.drop_out = nn.Dropout(args.drop_out)
        self.act_fn = ACT2FN[args.hidden_act]
    
    def forward(self, x: torch.Tensor):
        return self.drop_out(
                self.w_down(
                    self.act_fn(
                        self.w_gate(x)
                    ) * self.w_up(x)
                )
            )

class TransformerBlock(nn.Module):
    """
    Transformer 解码器块，采用 Pre-Norm 架构。

    【结构】
    ┌─────────────────────────────────────┐
    │           Input (x)                 │
    │               ↓                     │
    │  ┌─────────────────────────────┐    │
    │  │  RMSNorm → Attention        │────┼──→ + (残差连接)
    │  └─────────────────────────────┘    │         ↓
    │  ┌─────────────────────────────┐    │
    │  │  RMSNorm → FeedForward      │────┼──→ + (残差连接)
    │  └─────────────────────────────┘    │         ↓
    │                                     │      Output
    └─────────────────────────────────────┘

    【Pre-Norm vs Post-Norm】
    - Pre-Norm:  x + Sublayer(Norm(x))  ← 本实现采用
    - Post-Norm: Norm(x + Sublayer(x))  ← 原始 Transformer
    - Pre-Norm 训练更稳定，收敛更快，是现代 LLM 的标准选择
    """
    def __init__(self, layer_id: int, config: MortisConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.layer_id = layer_id

        # 自注意力层 (支持 GQA + Flash Attention)
        self.self_attn = Attention(config)

        # Pre-Norm: 在 Attention 之前归一化
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Pre-Norm: 在 FFN 之前归一化
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # 前馈网络 (SwiGLU 激活)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            freq_cis: RoPE 频率表 [seq_len, head_dim // 2]
            past_key_value: KV Cache (k, v)，用于推理加速
            use_cache: 是否返回更新后的 KV Cache
            attention_mask: padding mask [batch_size, seq_len]

        Returns:
            output: 输出张量 [batch_size, seq_len, hidden_size]
            present_key_value: 更新后的 KV Cache
        """
        # ========== 自注意力 + 残差连接 ==========
        # Pre-Norm: 先归一化再计算注意力
        residual = x
        h, present_key_value = self.self_attn(
            self.input_layernorm(x),
            freq_cis,
            past_key_value,
            use_cache,
            attention_mask
        )
        h = h + residual  # 残差连接

        # ========== 前馈网络 + 残差连接 ==========
        # Pre-Norm: 先归一化再计算 FFN
        h = h + self.mlp(self.post_attention_layernorm(h))

        return h, present_key_value
        
class MortisModel(nn.Module):
    """
    MiniMind 核心模型，包含 Embedding + N 层 TransformerBlock + 最终归一化。

    【结构】
    Input IDs → Embedding → Dropout → [TransformerBlock × N] → RMSNorm → Hidden States

    【特性】
    - 预计算 RoPE 频率表，支持 YaRN 长度外推
    - 支持 KV Cache 推理加速
    - 支持 padding mask
    """
    def __init__(self, config: MortisConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # Token Embedding: token_id → hidden_size 维向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Embedding Dropout: 防止过拟合
        self.drop_out = nn.Dropout(config.drop_out)

        # N 层 Transformer 解码器块
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id, config)
            for layer_id in range(config.num_hidden_layers)
        ])

        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 频率表 (复数形式)
        # shape: [max_position_embeddings, head_dim // 2]
        freq_cis = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        # 注册为 buffer，不参与梯度计算，但会随模型保存/加载
        self.register_buffer("freq_cis", freq_cis, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        **kwargs
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: padding mask [batch_size, seq_len], 1=有效, 0=padding
            past_key_values: 各层的 KV Cache 列表，用于推理加速
            use_cache: 是否返回更新后的 KV Cache

        Returns:
            hidden_states: 最终隐藏状态 [batch_size, seq_len, hidden_size]
            presents: 各层的 KV Cache 列表 (如果 use_cache=True)
        """
        batch_size, seq_len = input_ids.shape

        # 处理 HuggingFace 风格的 past_key_values (可能有 .layers 属性)
        if hasattr(past_key_values, "key_cache"):
            past_key_values = None

        # 初始化各层的 KV Cache 为 None
        past_key_values = past_key_values or [None] * self.num_hidden_layers

        # 计算起始位置 (用于 KV Cache 场景下的位置编码)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # ========== Embedding ==========
        hidden_states = self.embed_tokens(input_ids)  # [batch_size, seq_len, hidden_size]
        hidden_states = self.drop_out(hidden_states)

        # 获取当前位置对应的 RoPE 频率表
        freq_cis = self.freq_cis[start_pos: start_pos + seq_len]

        # ========== 逐层前向传播 ==========
        presents = []
        for layer_idx, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[layer_idx]
            hidden_states, present_kv = layer(
                hidden_states,
                freq_cis,
                past_key_value=layer_past_kv,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present_kv)

        # ========== 最终归一化 ==========
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents if use_cache else None
        

class MortisForCausalLM(PreTrainedModel, GenerationMixin):
    """
    因果语言模型 (Causal Language Model)，用于自回归文本生成。

    【结构】
    ┌─────────────────────────────────────────────────────────────┐
    │                   MortisForCausalLM                         │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │                    MortisModel                        │  │
    │  │  (Embedding → TransformerBlocks → RMSNorm)            │  │
    │  └───────────────────────────────────────────────────────┘  │
    │                          ↓                                  │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │       LM Head (权重与 Embedding 共享)                  │  │
    │  │  hidden_size → vocab_size (预测下一个 token)          │  │
    │  └───────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘

    【权重共享 (Weight Tying)】
    LM Head 与 Embedding 共享权重矩阵：
    - Embedding: [vocab_size, hidden_size] - 将 token 映射到向量
    - LM Head:   [hidden_size, vocab_size] - 将向量映射回 token 概率
    - 共享权重减少 ~30% 参数量，且语义上合理

    【继承】
    - PreTrainedModel: HuggingFace 基类，提供保存/加载、配置管理
    - GenerationMixin: 提供 generate() 方法 (采样、beam search 等)
    """
    config_class = MortisConfig

    def __init__(self, config: MortisConfig):
        self.config = config
        super().__init__(config)

        # 核心 Transformer 模型
        self.model = MortisModel(config)

        # LM Head: 将隐藏状态映射到词表概率分布
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        # 权重共享: LM Head 与 Embedding 使用同一权重矩阵
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: padding mask [batch_size, seq_len]
            past_key_values: KV Cache 列表
            use_cache: 是否返回 KV Cache
            logits_to_keep: 只计算最后 N 个位置的 logits (节省显存)
                - 0: 计算全部位置 (训练时)
                - 1: 只计算最后一个位置 (推理时)
                - N: 计算最后 N 个位置
                - Tensor: 自定义位置索引

        Returns:
            CausalLMOutputWithPast:
                - logits: 词表概率分布 [batch_size, seq_len or N, vocab_size]
                - past_key_values: KV Cache
                - hidden_states: 隐藏状态
        """
        # ========== 1. 前向传播获取隐藏状态 ==========
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        # ========== 2. 计算 logits (可选只计算部分位置) ==========
        # logits_to_keep 优化: 只对需要的位置计算 vocab 分布，节省显存
        # - 推理时只需最后一个位置: logits_to_keep=1
        # - 训练时需要全部位置: logits_to_keep=0
        # 注意: slice(-0, None) 等价于 slice(0, None)，即全部位置
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )











