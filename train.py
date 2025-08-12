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
    parser = argparse.ArgumentParser(description="BERT模型微调训练脚本")
    parser.add_argument("--model", required=True, help="预训练模型路径或模型名称")
    parser.add_argument("--train_file", required=True, help="训练数据文件 (jsonl格式)")
    parser.add_argument(
        "--validation_file", required=True, help="验证数据文件 (jsonl格式)"
    )
    parser.add_argument(
        "--label2id_file", required=True, help="标签映射文件 (json格式)"
    )
    parser.add_argument("--text_column_name", default="text", help="文本列名")
    parser.add_argument("--label_column_name", default="label", help="标签列名")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    args = parser.parse_args()

    # 单独创建 TrainingArguments，针对BERT优化
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,  # BERT通常可以使用更大的batch size
        per_device_eval_batch_size=16,
        learning_rate=2e-5,  # BERT推荐的学习率
        warmup_ratio=0.1,  # 10%的warmup步骤
        weight_decay=0.01,  # 权重衰减
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=True,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # 节省显存
        report_to=None,  # 不使用wandb等
    )

    # 1. 加载标签映射
    with open(args.label2id_file, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {id: label for label, id in label2id.items()}
    num_labels = len(label2id)

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
    
    print(f"加载的模型: {args.model}")
    print(f"标签数量: {num_labels}")
    print(f"标签映射: {label2id}")

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
        # 标签映射
        result["label"] = [label2id[l] for l in examples[args.label_column_name]]
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
