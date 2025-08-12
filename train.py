import json
import argparse
import torch
import os
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TensorBoardCallback(TrainerCallback):
    """自定义TensorBoard回调，用于记录更多的训练指标"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = None
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"TensorBoard 日志目录: {self.log_dir}")
            print(f"启动TensorBoard命令: tensorboard --logdir={self.log_dir}")
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self.writer and logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, state.global_step)
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if self.writer and metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key.startswith('eval_'):
                    self.writer.add_scalar(f"evaluation/{key}", value, state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.writer:
            self.writer.close()
            print("TensorBoard 记录已保存")


def main():
    parser = argparse.ArgumentParser(description="BERT模型微调训练脚本")
    parser.add_argument("--model", required=True, help="预训练模型路径或模型名称")
    parser.add_argument("--train_file", required=True, help="训练数据文件 (jsonl格式)")
    parser.add_argument(
        "--validation_file", required=True, help="验证数据文件 (jsonl格式)"
    )
    parser.add_argument("--text_column_name", default="text", help="文本列名")
    parser.add_argument("--label_column_name", default="valid", help="标签列名")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    
    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help="训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None, help="评估批次大小")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志记录步数")
    parser.add_argument("--save_total_limit", type=int, default=2, help="保存模型数量限制")
    
    # 设备和性能参数
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="训练设备")
    parser.add_argument("--gpus", default="0", help="使用的GPU ID（逗号分隔）")
    parser.add_argument("--fp16", action="store_true", help="使用混合精度训练")
    parser.add_argument("--no_fp16", action="store_true", help="禁用混合精度训练")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="禁用梯度检查点")
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="启用数据加载器内存固定")
    
    # TensorBoard参数
    parser.add_argument("--tensorboard_log_dir", default=None, help="TensorBoard日志目录")
    parser.add_argument("--no_tensorboard", action="store_true", help="禁用TensorBoard记录")
    
    args = parser.parse_args()

    # 设备选择逻辑
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("警告: 指定使用CUDA但未检测到GPU，将使用CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 批次大小设置
    if args.per_device_train_batch_size is None:
        train_batch_size = 16 if device.type == "cuda" else 4
    else:
        train_batch_size = args.per_device_train_batch_size
    
    if args.per_device_eval_batch_size is None:
        eval_batch_size = train_batch_size
    else:
        eval_batch_size = args.per_device_eval_batch_size

    # FP16设置
    if args.no_fp16:
        use_fp16 = False
    elif args.fp16:
        use_fp16 = True
    else:
        use_fp16 = device.type == "cuda"  # 默认GPU使用FP16，CPU不使用

    # 梯度检查点设置
    if args.no_gradient_checkpointing:
        use_gradient_checkpointing = False
    elif args.gradient_checkpointing:
        use_gradient_checkpointing = True
    else:
        use_gradient_checkpointing = device.type == "cuda"  # 默认GPU使用梯度检查点

    # 数据加载器内存固定
    use_pin_memory = args.dataloader_pin_memory or (device.type == "cuda")

    print(f"训练批次大小: {train_batch_size}")
    print(f"评估批次大小: {eval_batch_size}")
    print(f"使用FP16: {use_fp16}")
    print(f"梯度检查点: {use_gradient_checkpointing}")
    print(f"内存固定: {use_pin_memory}")

    # TensorBoard设置
    use_tensorboard = not args.no_tensorboard
    if use_tensorboard:
        if args.tensorboard_log_dir:
            tensorboard_log_dir = args.tensorboard_log_dir
        else:
            tensorboard_log_dir = os.path.join(args.output_dir, "tensorboard_logs")
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        print(f"TensorBoard日志目录: {tensorboard_log_dir}")
        print(f"启动TensorBoard命令: tensorboard --logdir={tensorboard_log_dir}")
    else:
        tensorboard_log_dir = None
        print("TensorBoard记录已禁用")

    # 单独创建 TrainingArguments，针对BERT优化
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=args.save_total_limit,
        fp16=use_fp16,
        dataloader_pin_memory=use_pin_memory,
        gradient_checkpointing=use_gradient_checkpointing,
        report_to="tensorboard" if use_tensorboard else None,
        logging_dir=tensorboard_log_dir if use_tensorboard else None,
        logging_first_step=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
    )

    # 1. 创建二分类标签映射（valid字段是布尔值）
    label2id = {"false": 0, "true": 1}  # 将布尔值映射为整数
    id2label = {0: "false", 1: "true"}
    num_labels = 2  # 二分类任务

    # 2. 加载BERT模型和分词器
    config = BertConfig.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    
    # 使用BertTokenizer，确保兼容性
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=True)
    
    # 使用BertForSequenceClassification进行分类任务
    model = BertForSequenceClassification.from_pretrained(
        args.model,
        config=config,
    )
    
    # 将模型移动到正确的设备
    model.to(device)
    
    print(f"加载的模型: {args.model}")
    print(f"标签数量: {num_labels}")
    print(f"标签映射: {label2id}")
    print(f"模型设备: {next(model.parameters()).device}")

    # 3. 加载和预处理数据集
    datasets = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.validation_file},
    )

    def preprocess_function(examples):
        # BERT分词，添加特殊tokens
        result = tokenizer(
            examples[args.text_column_name],
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=True,  # 添加[CLS]和[SEP]
            return_attention_mask=True,  # 返回attention mask
        )
        # 将布尔值转换为整数标签
        result["label"] = [1 if val else 0 for val in examples[args.label_column_name]]
        return result

    processed_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="正在处理数据集",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")

    # 4. 设置训练器，使用BERT特定的数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # 定义评估指标
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # 计算准确率
        accuracy = np.mean(predictions == labels)
        
        # 计算精确率、召回率和F1分数
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
        }

    # 准备回调函数
    callbacks = []
    if use_tensorboard:
        tb_callback = TensorBoardCallback(tensorboard_log_dir)
        callbacks.append(tb_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 5. 训练
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 6. 评估
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print("训练和评估完成!")
    if use_tensorboard:
        print(f"\n=== TensorBoard 可视化 ===")
        print(f"TensorBoard 日志保存在: {tensorboard_log_dir}")
        print(f"启动 TensorBoard 查看训练过程:")
        print(f"  tensorboard --logdir={tensorboard_log_dir}")
        print(f"然后在浏览器中访问: http://localhost:6006")
        print(f"=========================\n")


if __name__ == "__main__":
    main()
