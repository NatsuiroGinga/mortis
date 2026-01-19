#!/bin/bash

#########################################################################
# File Name:    pretrain.sh
# Author:       mortiswang
# mail:         mortiswang@tencent.com
# Created Time: Mon 19 Jan 2026 10:31:45 AM CST
# Description:  Mortis 预训练启动脚本 - 支持14G和24G显存模式
#########################################################################

set -e  # 遇到错误立即退出

# 脚本参数解析
usage() {
    echo "用法: $0 [模式] [选项]"
    echo ""
    echo "模式:"
    echo "  14g  - 14G显存模式 (适合RTX 4060 Ti 16G等显卡)"
    echo "  24g  - 24G显存模式 (适合RTX 3090/4090等显卡)"
    echo ""
    echo "选项:"
    echo "  -d, --data-path PATH    数据文件路径 (默认: ../dataset/pretrain_hq.jsonl)"
    echo "  -e, --epochs NUM        训练轮数 (默认: 1)"
    echo "  -l, --learning-rate LR  学习率 (默认: 5e-4)"
    echo "  -s, --save-dir DIR     模型保存目录 (默认: ../out)"
    echo "  -w, --wandb            启用wandb日志记录"
    echo "  -h, --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 14g -e 2 -w                    # 14G模式，训练2轮，启用wandb"
    echo "  $0 24g -d /path/to/data.jsonl      # 24G模式，自定义数据路径"
}

# 默认参数
MODE=""
DATA_PATH="./dataset/pretrain_hq.jsonl"
EPOCHS=1
LEARNING_RATE="5e-4"
SAVE_DIR="../out"
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

# 检查模式参数
if [[ -z "$MODE" ]]; then
    echo "错误: 必须指定模式 (14g 或 24g)"
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

# 显示配置信息
echo "=========================================="
echo " Mortis 预训练配置"
echo "=========================================="
echo "模式:              $MODE"
echo "数据路径:          $DATA_PATH"
echo "训练轮数:          $EPOCHS"
echo "学习率:            $LEARNING_RATE"
echo "保存目录:          $SAVE_DIR"
echo "设备:              $DEVICE"
echo "WandB日志:         $([ $USE_WANDB -eq 1 ] && echo "启用" || echo "禁用")"
echo "=========================================="

# 根据模式设置不同的训练参数
case $MODE in
    14g)
        # 14G显存模式 - 占满14G显存 (如NVIDIA L20)
        echo "使用14G显存模式配置..."
        BATCH_SIZE=40
        ACCUMULATION_STEPS=1
        HIDDEN_SIZE=768
        NUM_LAYERS=12
        MAX_SEQ_LEN=512
        GRAD_CLIP=1.0
        ;;
    24g)
        # 24G显存模式 - 适合RTX 3090/4090等显卡
        echo "使用24G显存模式配置..."
        BATCH_SIZE=32
        ACCUMULATION_STEPS=2
        HIDDEN_SIZE=768
        NUM_LAYERS=16
        MAX_SEQ_LEN=1024
        GRAD_CLIP=1.0
        ;;
    *)
        echo "错误: 未知模式 '$MODE'"
        exit 1
        ;;
esac

# 构建训练命令
TRAIN_CMD="python trainer/train_pretrain.py \
    --data_path \"$DATA_PATH\" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --save_dir \"$SAVE_DIR\" \
    --device $DEVICE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --grad_clip $GRAD_CLIP \
    --hidden_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_LAYERS \
    --max_seq_len $MAX_SEQ_LEN \
    --log_interval 50 \
    --save_interval 500"

# 添加wandb选项
if [[ $USE_WANDB -eq 1 ]]; then
    TRAIN_CMD="$TRAIN_CMD --use_wandb"
fi

# 显示训练配置详情
echo ""
echo "训练配置详情:"
echo "- Batch Size:        $BATCH_SIZE"
echo "- 梯度累积步数:      $ACCUMULATION_STEPS"
echo "- 有效Batch Size:   $(($BATCH_SIZE * $ACCUMULATION_STEPS))"
echo "- 隐藏层维度:        $HIDDEN_SIZE"
echo "- Transformer层数:  $NUM_LAYERS"
echo "- 最大序列长度:      $MAX_SEQ_LEN"
echo "- 梯度裁剪:          $GRAD_CLIP"
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
echo "开始训练..."
echo "命令: $TRAIN_CMD"
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
if [[ -f "$SAVE_DIR/pretrain_${HIDDEN_SIZE}.pth" ]]; then
    echo "模型文件: pretrain_${HIDDEN_SIZE}.pth"
    if command -v python &> /dev/null; then
        python -c "
import torch
model_path = '$SAVE_DIR/pretrain_${HIDDEN_SIZE}.pth'
state_dict = torch.load(model_path, map_location='cpu')
print(f'模型参数量: {sum(p.numel() for p in state_dict.values()) / 1e6:.2f}M')
print(f'模型大小: {sum(p.numel() * p.element_size() for p in state_dict.values()) / 1024 / 1024:.2f}MB')
"
    fi
fi
