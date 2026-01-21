#!/bin/bash

#########################################################################
# File Name:    eval.sh
# Author:       mortiswang
# mail:         mortiswang@tencent.com
# Created Time: Mon 20 Jan 2026 10:00:00 AM CST
# Description:  Mortis 模型推理验证脚本 - 支持14G和24G显存模式
#########################################################################

set -e  # 遇到错误立即退出

# 脚本参数解析
usage() {
    echo "用法: $0 <模式> [选项]"
    echo ""
    echo "模式 (必填):"
    echo "  14g  - 14G显存模式 (hidden_size=768, num_layers=12)"
    echo "  24g  - 24G显存模式 (hidden_size=768, num_layers=16)"
    echo ""
    echo "选项:"
    echo "  -w, --weight TYPE       权重类型 (默认: full_sft)"
    echo "                          可选: pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo"
    echo "  -s, --save-dir DIR      模型权重目录 (默认: ./out)"
    echo "  -t, --temperature NUM   生成温度 (默认: 0.85)"
    echo "  -p, --top-p NUM         nucleus采样阈值 (默认: 0.85)"
    echo "  -m, --max-tokens NUM    最大生成长度 (默认: 8192)"
    echo "  -i, --interactive       手动输入模式 (默认: 自动测试)"
    echo "  -r, --rope-scaling      启用RoPE位置编码外推 (16倍外推)"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 14g                      # 14G模式，自动测试"
    echo "  $0 24g -i                   # 24G模式，手动输入"
    echo "  $0 14g -w pretrain          # 14G模式，使用pretrain权重"
    echo "  $0 14g -t 0.7 -p 0.9        # 自定义生成参数"
}

# 默认参数
MODE=""
WEIGHT="full_sft"
SAVE_DIR="../out"
TEMPERATURE="0.85"
TOP_P="0.85"
MAX_TOKENS="512"
INTERACTIVE=0
ROPE_SCALING=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        14g|24g)
            MODE=$1
            shift
            ;;
        -w|--weight)
            WEIGHT="$2"
            shift 2
            ;;
        -s|--save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -p|--top-p)
            TOP_P="$2"
            shift 2
            ;;
        -m|--max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -i|--interactive)
            INTERACTIVE=1
            shift
            ;;
        -r|--rope-scaling)
            ROPE_SCALING=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "错误: 未知参数 '$1'"
            usage
            exit 1
            ;;
    esac
done

# 检查模式参数 (必填)
if [[ -z "$MODE" ]]; then
    echo "错误: 必须指定模式 (14g 或 24g)"
    echo ""
    usage
    exit 1
fi

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请确保已安装Python 3.8+"
    exit 1
fi

# 检查CUDA是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: 未检测到NVIDIA GPU，将使用CPU模式"
    DEVICE="cpu"
else
    DEVICE="cuda"
fi

# 根据模式设置模型参数
case $MODE in
    14g)
        HIDDEN_SIZE=768
        NUM_LAYERS=12
        ;;
    24g)
        HIDDEN_SIZE=768
        NUM_LAYERS=16
        ;;
    *)
        echo "错误: 未知模式 '$MODE'"
        exit 1
        ;;
esac

# 显示配置信息
echo "=========================================="
echo " Mortis 模型推理配置"
echo "=========================================="
echo "模式:              $MODE"
echo "权重类型:          $WEIGHT"
echo "权重目录:          $SAVE_DIR"
echo "设备:              $DEVICE"
echo "隐藏层维度:        $HIDDEN_SIZE"
echo "Transformer层数:   $NUM_LAYERS"
echo "生成温度:          $TEMPERATURE"
echo "Top-P:             $TOP_P"
echo "最大生成长度:      $MAX_TOKENS"
echo "输入模式:          $([ $INTERACTIVE -eq 1 ] && echo "手动输入" || echo "自动测试")"
echo "RoPE外推:          $([ $ROPE_SCALING -eq 1 ] && echo "启用" || echo "禁用")"
echo "=========================================="
echo ""

# 构建推理命令
EVAL_CMD="python eval_llm.py \
    --load_from model \
    --save_dir \"$SAVE_DIR\" \
    --weight $WEIGHT \
    --hidden_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_LAYERS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_new_tokens $MAX_TOKENS \
    --device $DEVICE"

# 添加RoPE外推选项
if [[ $ROPE_SCALING -eq 1 ]]; then
    EVAL_CMD="$EVAL_CMD --inference_rope_scaling"
fi

# 显示命令
echo "执行命令:"
echo "$EVAL_CMD"
echo ""

# 执行推理命令
if [[ $INTERACTIVE -eq 1 ]]; then
    echo "启动手动输入模式..."
    eval $EVAL_CMD
else
    echo "启动自动测试模式..."
    echo "0" | eval $EVAL_CMD
fi
