import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


def main():
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: 未检测到GPU，将使用CPU训练（速度会很慢）")
    
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
    args = parser.parse_args()

    # 单独创建 TrainingArguments，针对BERT优化
    # 根据GPU可用性调整参数
    if torch.cuda.is_available():
        batch_size = 16
        use_fp16 = True
        gradient_checkpointing = True
    else:
        batch_size = 4  # CPU时使用较小的batch size
        use_fp16 = False  # CPU不支持fp16
        gradient_checkpointing = False
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,  # BERT推荐的学习率
        warmup_ratio=0.1,  # 10%的warmup步骤
        weight_decay=0.01,  # 权重衰减
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=use_fp16,
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_checkpointing=gradient_checkpointing,
        report_to=None,  # 不使用wandb等
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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


if __name__ == "__main__":
    main()
