#!/bin/bash

#########################################################################
# File Name:    full_sft.sh
# Author:       mortiswang
# mail:         mortiswang@tencent.com
# Created Time: Mon 20 Jan 2026 11:30:00 AM CST
# Description:  Mortis Full SFT 启动脚本 - 支持14G和24G显存模式
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
    echo "  -d, --data-path PATH    SFT数据文件路径 (默认: ../dataset/sft_mini_512.jsonl)"
    echo "  -e, --epochs NUM        训练轮数 (默认: 2)"
    echo "  -l, --learning-rate LR  学习率 (默认: 1e-6)"
    echo "  -s, --save-dir DIR      模型保存目录 (默认: ../out)"
    echo "  -f, --from-weight NAME  基础权重名称 (默认: pretrain, 可选: none表示从头训练)"
    echo "  -r, --resume            启用断点续训"
    echo "  -w, --wandb             启用wandb日志记录"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 14g                        # 14G模式，基于pretrain权重训练"
    echo "  $0 14g -e 3 -w                # 14G模式，训练3轮，启用wandb"
    echo "  $0 14g -f none                # 14G模式，从头训练（不加载pretrain）"
    echo "  $0 24g -d /path/to/sft.jsonl  # 24G模式，自定义数据路径"
    echo "  $0 14g -r -w                  # 断点续训 + wandb"
}

# 默认参数
MODE=""
DATA_PATH="./dataset/sft_mini_512.jsonl"
EPOCHS=2
LEARNING_RATE="1e-6"
SAVE_DIR="../out"
FROM_WEIGHT="pretrain"
FROM_RESUME=0
USE_WANDB=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        14g|24g)
            MODE=$1
            shift
            ;;
        -d|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -s|--save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        -f|--from-weight)
            FROM_WEIGHT="$2"
            shift 2
            ;;
        -r|--resume)
            FROM_RESUME=1
            shift
            ;;
        -w|--wandb)
            USE_WANDB=1
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

# 检查数据文件是否存在
if [[ ! -f "$DATA_PATH" ]]; then
    echo "错误: 数据文件 '$DATA_PATH' 不存在"
    echo "请确保数据文件存在或使用 -d 参数指定正确的路径"
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
    echo "检测到NVIDIA GPU，使用CUDA加速"
fi

# 根据模式设置模型参数
case $MODE in
    14g)
        HIDDEN_SIZE=768
        NUM_LAYERS=12
        BATCH_SIZE=16
        ACCUMULATION_STEPS=1
        MAX_SEQ_LEN=512
        ;;
    24g)
        HIDDEN_SIZE=768
        NUM_LAYERS=16
        BATCH_SIZE=16
        ACCUMULATION_STEPS=2
        MAX_SEQ_LEN=1024
        ;;
    *)
        echo "错误: 未知模式 '$MODE'"
        exit 1
        ;;
esac

# 显示配置信息
echo "=========================================="
echo " Mortis Full SFT 训练配置"
echo "=========================================="
echo "模式:              $MODE"
echo "数据路径:          $DATA_PATH"
echo "训练轮数:          $EPOCHS"
echo "学习率:            $LEARNING_RATE"
echo "保存目录:          $SAVE_DIR"
echo "设备:              $DEVICE"
echo "基础权重:          $FROM_WEIGHT"
echo "断点续训:          $([ $FROM_RESUME -eq 1 ] && echo "启用" || echo "禁用")"
echo "WandB日志:         $([ $USE_WANDB -eq 1 ] && echo "启用" || echo "禁用")"
echo "=========================================="
echo ""
echo "模型配置详情:"
echo "- Batch Size:        $BATCH_SIZE"
echo "- 梯度累积步数:      $ACCUMULATION_STEPS"
echo "- 有效Batch Size:    $(($BATCH_SIZE * $ACCUMULATION_STEPS))"
echo "- 隐藏层维度:        $HIDDEN_SIZE"
echo "- Transformer层数:   $NUM_LAYERS"
echo "- 最大序列长度:      $MAX_SEQ_LEN"
echo "=========================================="
echo ""

# 构建训练命令
TRAIN_CMD="python trainer/train_full_sft.py \
    --data_path \"$DATA_PATH\" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --save_dir \"$SAVE_DIR\" \
    --device $DEVICE \
    --dtype bfloat16 \
    --accumulation_steps $ACCUMULATION_STEPS \
    --grad_clip 1.0 \
    --hidden_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_LAYERS \
    --max_seq_len $MAX_SEQ_LEN \
    --log_interval 100 \
    --save_interval 500 \
    --from_weight $FROM_WEIGHT \
    --from_resume $FROM_RESUME"

# 添加wandb选项
if [[ $USE_WANDB -eq 1 ]]; then
    TRAIN_CMD="$TRAIN_CMD --use_wandb"
fi

# 显示命令
echo "执行命令:"
echo "$TRAIN_CMD"
echo ""

# 确认是否开始训练
read -p "是否开始训练? (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 开始训练
echo "开始 Full SFT 训练..."
echo ""

# 执行训练命令
eval $TRAIN_CMD

# 训练完成
echo ""
echo "=========================================="
echo " 训练完成!"
echo " 模型已保存到: $SAVE_DIR"
echo "=========================================="

# 显示模型信息
if [[ -f "$SAVE_DIR/full_sft_${HIDDEN_SIZE}.pth" ]]; then
    echo "模型文件: full_sft_${HIDDEN_SIZE}.pth"
    if command -v python &> /dev/null; then
        python -c "
import torch
model_path = '$SAVE_DIR/full_sft_${HIDDEN_SIZE}.pth'
state_dict = torch.load(model_path, map_location='cpu')
print(f'模型参数量: {sum(p.numel() for p in state_dict.values()) / 1e6:.2f}M')
print(f'模型大小: {sum(p.numel() * p.element_size() for p in state_dict.values()) / 1024 / 1024:.2f}MB')
"
    fi
fi
