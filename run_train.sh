#!/bin/bash

# BERT模型训练脚本 - Linux版本
# 使用方法: ./run_train.sh [选项]

set -e  # 遇到错误时退出

# 默认参数
MODEL_PATH="./model"
TRAIN_FILE="./dataset/train.jsonl"
VALIDATION_FILE="./dataset/val.jsonl"
OUTPUT_DIR="./outputs/bert-classification"
BATCH_SIZE=""
EPOCHS=""
LEARNING_RATE=""
DEVICE="auto"
GPU_ID=""
EXTRA_ARGS=""

# 显示使用方法
show_usage() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -m, --model PATH        预训练模型路径 (默认: ./model)"
    echo "  -t, --train FILE        训练数据文件 (默认: ./dataset/train.jsonl)"
    echo "  -v, --validation FILE   验证数据文件 (默认: ./dataset/val.jsonl)"
    echo "  -o, --output DIR        输出目录 (默认: ./outputs/bert-classification)"
    echo "  -b, --batch-size N      批次大小"
    echo "  -e, --epochs N          训练轮数"
    echo "  -l, --learning-rate N   学习率"
    echo "  -d, --device DEVICE     设备选择 (auto/cpu/cuda)"
    echo "  -g, --gpu-id N          指定GPU ID (0,1,2...)"
    echo "  --fp16                  启用混合精度训练"
    echo "  --cpu                   强制使用CPU训练"
    echo "  --gradient-checkpointing 启用梯度检查点"
    echo ""
    echo "示例:"
    echo "  $0                                    # 默认配置"
    echo "  $0 --gpu-id 0 --batch-size 16       # 使用GPU 0，批次大小16"
    echo "  $0 --cpu --batch-size 4              # CPU训练，批次大小4"
    echo "  $0 --epochs 5 --learning-rate 3e-5   # 5轮训练，自定义学习率"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -t|--train)
            TRAIN_FILE="$2"
            shift 2
            ;;
        -v|--validation)
            VALIDATION_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
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
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -g|--gpu-id)
            GPU_ID="$2"
            DEVICE="cuda"
            shift 2
            ;;
        --fp16)
            EXTRA_ARGS="$EXTRA_ARGS --fp16"
            shift
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --gradient-checkpointing)
            EXTRA_ARGS="$EXTRA_ARGS --gradient_checkpointing"
            shift
            ;;
        *)
            echo "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 检查Python虚拟环境
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "使用虚拟环境: $VIRTUAL_ENV"
elif [[ -f ".venv/bin/activate" ]]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
else
    echo "警告: 未检测到虚拟环境"
fi

# 检查必要文件
echo "检查必要文件..."
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "错误: 模型目录不存在: $MODEL_PATH"
    echo "请先运行: python download_model.py"
    exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "错误: 训练文件不存在: $TRAIN_FILE"
    echo "请先运行: python split_dataset.py"
    exit 1
fi

if [[ ! -f "$VALIDATION_FILE" ]]; then
    echo "错误: 验证文件不存在: $VALIDATION_FILE"
    echo "请先运行: python split_dataset.py"
    exit 1
fi

# 设置GPU环境变量
if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "设置CUDA_VISIBLE_DEVICES=$GPU_ID"
fi

# 检查GPU可用性
if [[ "$DEVICE" == "cuda" || "$DEVICE" == "auto" ]]; then
    python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"
fi

# 构建训练命令
CMD="python train.py"
CMD="$CMD --model $MODEL_PATH"
CMD="$CMD --train_file $TRAIN_FILE"
CMD="$CMD --validation_file $VALIDATION_FILE"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --device $DEVICE"

if [[ -n "$BATCH_SIZE" ]]; then
    CMD="$CMD --per_device_train_batch_size $BATCH_SIZE"
fi

if [[ -n "$EPOCHS" ]]; then
    CMD="$CMD --num_train_epochs $EPOCHS"
fi

if [[ -n "$LEARNING_RATE" ]]; then
    CMD="$CMD --learning_rate $LEARNING_RATE"
fi

CMD="$CMD $EXTRA_ARGS"

# 显示训练配置
echo ""
echo "=================================="
echo "BERT模型训练配置"
echo "=================================="
echo "模型路径: $MODEL_PATH"
echo "训练文件: $TRAIN_FILE"
echo "验证文件: $VALIDATION_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "设备: $DEVICE"
if [[ -n "$GPU_ID" ]]; then
    echo "GPU ID: $GPU_ID"
fi
if [[ -n "$BATCH_SIZE" ]]; then
    echo "批次大小: $BATCH_SIZE"
fi
if [[ -n "$EPOCHS" ]]; then
    echo "训练轮数: $EPOCHS"
fi
if [[ -n "$LEARNING_RATE" ]]; then
    echo "学习率: $LEARNING_RATE"
fi
echo "额外参数: $EXTRA_ARGS"
echo "完整命令: $CMD"
echo "=================================="
echo ""

# 开始训练
echo "开始训练..."
eval $CMD

echo ""
echo "训练完成!"
echo "模型保存在: $OUTPUT_DIR"
