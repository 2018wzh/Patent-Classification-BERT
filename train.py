import json
import argparse
import torch
import os
from typing import Any, List, Dict
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm


class JsonlClassificationDataset(torch.utils.data.Dataset):
    """读取预先分词(tokenized)好的 jsonl 文件，支持 memory / stream 策略与自动回退。

    strategy:
      - memory: 读取全部行到内存后解析 (速度快，需足够内存)
      - stream: 两遍文件扫描 (第一遍统计非空行数，第二遍解析)；内存占用低
    自动回退: memory 模式若发生 OSError 或 MemoryError，则回退到 stream。
    """

    def __init__(self, path: str, show_progress: bool = True, strategy: str = "memory"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到数据文件: {path}")
        self.path = path
        self.samples: List[Dict[str, Any]] = []
        strategy = (strategy or "memory").lower()
        if strategy not in {"memory", "stream"}:
            print(f"[Dataset] 未知策略 {strategy} 使用 memory")
            strategy = "memory"
        try:
            if strategy == "memory":
                self._load_memory(show_progress)
            else:
                self._load_stream(show_progress)
        except (OSError, MemoryError) as e:
            print(f"[Dataset] memory 策略失败: {e}. 回退 stream 模式。")
            self.samples.clear()
            self._load_stream(show_progress)

    # ====== 内部方法 ======
    def _normalize(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        label_val = obj.get("label") if "label" in obj else obj.get("labels")
        if isinstance(label_val, bool):
            label_val = 1 if label_val else 0
        return {
            "input_ids": obj["input_ids"],
            "attention_mask": obj["attention_mask"],
            "labels": int(label_val) if label_val is not None else 0,
        }

    def _load_memory(self, show_progress: bool):
        size_mb = os.path.getsize(self.path) / 1024 / 1024
        print(f"[Dataset] memory 模式加载 {self.path} ({size_mb:.2f} MB)")
        with open(self.path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        lines = [ln for ln in lines if ln.strip()]
        print(f"[Dataset] 非空行: {len(lines)}")
        it = lines
        if show_progress:
            it = tqdm(lines, desc=f"解析 {os.path.basename(self.path)}", unit="行")  # type: ignore
        bad = 0
        for line in it:  # type: ignore
            try:
                obj = json.loads(line)
                self.samples.append(self._normalize(obj))
            except Exception:
                bad += 1
        if bad:
            print(f"[Dataset] 跳过损坏行: {bad}")

    def _load_stream(self, show_progress: bool):
        size_mb = os.path.getsize(self.path) / 1024 / 1024
        print(f"[Dataset] stream 模式加载 {self.path} ({size_mb:.2f} MB)")
        # 第一遍统计
        with open(self.path, "r", encoding="utf-8") as f:
            total = sum(1 for ln in f if ln.strip())
        print(f"[Dataset] 预计解析行数: {total}")
        # 第二遍解析
        with open(self.path, "r", encoding="utf-8") as f:
            iterator = f
            if show_progress:
                iterator = tqdm(f, total=total, desc=f"解析 {os.path.basename(self.path)}", unit="行")  # type: ignore
            bad = 0
            for line in iterator:  # type: ignore
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self.samples.append(self._normalize(obj))
                except Exception:
                    bad += 1
            if bad:
                print(f"[Dataset] 跳过损坏行: {bad}")

    # ====== Dataset 接口 ======
    def __len__(self):  # type: ignore
        return len(self.samples)

    def __getitem__(self, idx):  # type: ignore
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(s["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(s["labels"], dtype=torch.long),
        }


class PreTokenizedCollator:
    def __init__(self, pad_token_id: int = 0, max_seq_length: int | None = None, truncate: bool = True):
        """动态 padding 的同时可选截断。

        max_seq_length: 若提供则批内最长不超过该值，超过将截断 (truncate=True) 或报错 (truncate=False)
        """
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.truncate = truncate

    def __call__(self, features: List[Dict[str, torch.Tensor]]):
        # 计算批内原始最大长度
        raw_max = max(f["input_ids"].size(0) for f in features)
        if self.max_seq_length is not None and raw_max > self.max_seq_length:
            if not self.truncate:
                raise ValueError(f"批次中出现长度 {raw_max} > 允许最大 {self.max_seq_length}")
            target_max = self.max_seq_length
        else:
            target_max = raw_max
        ids_batch, mask_batch, label_batch = [], [], []
        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            if ids.size(0) > target_max:
                # 截断
                ids = ids[:target_max]
                mask = mask[:target_max]
            pad_len = target_max - ids.size(0)
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=self.pad_token_id)
                mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
            ids_batch.append(ids)
            mask_batch.append(mask)
            label_batch.append(f["labels"])
        return {
            "input_ids": torch.stack(ids_batch),
            "attention_mask": torch.stack(mask_batch),
            "labels": torch.stack(label_batch),
        }


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
                if isinstance(value, (int, float)) and key.startswith("eval_"):
                    self.writer.add_scalar(
                        f"evaluation/{key}", value, state.global_step
                    )

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer:
            self.writer.close()
            print("TensorBoard 记录已保存")


def main():
    parser = argparse.ArgumentParser(description="BERT模型微调训练脚本")
    # 新的配置文件使用统一的 config.json，其中包含 trainConfig 节点
    parser.add_argument(
        "--config_file",
        default="config/config.json",
        help="配置文件路径 (包含 trainConfig 节点)",
    )
    cli_args = parser.parse_args()

    # 从JSON文件加载配置，并取得 trainConfig 部分
    with open(cli_args.config_file, "r", encoding="utf-8") as f:
        full_config = json.load(f)

    if "trainConfig" not in full_config:
        raise ValueError("配置文件缺少 'trainConfig' 节点，请检查 config.json 结构。")

    train_cfg_dict = full_config["trainConfig"]
    pack_cfg = full_config.get('packConfig') or {}

    # 为了方便访问，将字典转换为对象（不考虑旧格式兼容）
    class ConfigObject:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    args: Any = ConfigObject(train_cfg_dict)  # 简单处理，避免类型检查器属性报错
    print(f"加载配置文件: {cli_args.config_file}")
    print(f"trainConfig 字段: {list(train_cfg_dict.keys())}")

    # 设置可见的GPU
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"设置 CUDA_VISIBLE_DEVICES='{args.gpus}'")

    # 自动检测设备并打印信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    strategy = args.dataset_strategy
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f"可用GPU数量: {num_gpus}")
        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - 显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
            )

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
    use_fp16 = device.type == "cuda"  # 默认GPU使用FP16，CPU不使用

    # 梯度检查点设置
    use_gradient_checkpointing = device.type == "cuda"  # 默认GPU使用梯度检查点

    # 数据加载器内存固定
    use_pin_memory = args.dataloader_pin_memory or (device.type == "cuda")

    print(f"训练批次大小: {train_batch_size}")
    print(f"评估批次大小: {eval_batch_size}")
    print(f"使用FP16: {use_fp16}")
    print(f"梯度检查点: {use_gradient_checkpointing}")
    print(f"内存固定: {use_pin_memory}")

    # TensorBoard设置
    tensorboard_log_dir = os.path.join(args.output_dir, "tensorboard_logs")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    print(f"TensorBoard日志目录: {tensorboard_log_dir}")
    print(f"启动TensorBoard命令: tensorboard --logdir={tensorboard_log_dir}")

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
        eval_strategy="epoch",  # 兼容当前 transformers 版本使用的参数名
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=args.save_total_limit,
        fp16=use_fp16,
        dataloader_pin_memory=use_pin_memory,
        gradient_checkpointing=use_gradient_checkpointing,
        report_to="tensorboard",
        logging_dir=tensorboard_log_dir,
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

    print(f"加载的模型: {args.model}")
    print(f"标签数量: {num_labels}")
    print(f"标签映射: {label2id}")

    # 模型最大位置长度
    model_max_pos = getattr(config, 'max_position_embeddings', None)
    if model_max_pos is None:
        model_max_pos = 512  # 兜底
    desired_train_max = getattr(args, 'max_seq_length', model_max_pos)
    pack_desired = pack_cfg.get('max_seq_length') if pack_cfg else None
    effective_seq_len = min([v for v in [model_max_pos, desired_train_max, pack_desired] if isinstance(v, int)]) if any(isinstance(v,int) for v in [model_max_pos, desired_train_max, pack_desired]) else model_max_pos
    print(f"模型 max_position_embeddings={model_max_pos} 期望 max_seq_length={desired_train_max} pack_max={pack_desired} -> 有效训练长度={effective_seq_len}")

    # 3. 数据集加载: 优先使用预打包 pt -> 其次 jsonl
    class PackedTensorDataset(torch.utils.data.Dataset):
        def __init__(self, pt_path: str):
            obj = torch.load(pt_path, map_location='cpu')
            self.input_ids = obj['input_ids']
            self.attention_mask = obj['attention_mask']
            self.labels = obj['labels']
            self.meta = obj.get('meta', {})
        def __len__(self):
            return self.input_ids.size(0)
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx],
            }

    def pick_dataset(path: str, split: str):
        if path.endswith('.pt') and os.path.exists(path):
            ds = PackedTensorDataset(path)
            print(f"[{split}] 直接加载打包数据: {path} shape={tuple(ds.input_ids.shape)}")
            return ds, True
        if path.endswith('.jsonl'):
            packed_alt = path[:-6] + '_packed.pt'
            if os.path.exists(packed_alt):
                ds = PackedTensorDataset(packed_alt)
                print(f"[{split}] 自动发现打包文件 {packed_alt} shape={tuple(ds.input_ids.shape)}")
                return ds, True
        # fallback jsonl
        ds = JsonlClassificationDataset(path, strategy=strategy)
        print(f"[{split}] 使用 jsonl ({path}) size={len(ds)} strategy={strategy}")
        return ds, False

    # 若 packConfig.enable 且未找到现成打包文件，可提示用户先执行 pack_dataset
    train_dataset, train_packed = pick_dataset(args.train_file, 'train')
    eval_dataset, eval_packed = pick_dataset(args.validation_file, 'val')
    if pack_cfg.get('enable') and (not train_packed or not eval_packed):
        print('[提示] packConfig.enable=true 但未检测到全部打包文件。可运行:')
        print('  python pack_dataset.py --inputs {0} {1} --out_dir {2} --max_seq_length {3}'.format(
            args.train_file,
            args.validation_file,
            os.path.dirname(args.train_file) or 'dataset',
            pack_cfg.get('max_seq_length', args.max_seq_length if hasattr(args,'max_seq_length') else 512)
        ))
    print(f"训练集大小: {len(train_dataset)}  验证集大小: {len(eval_dataset)}")

    # 4. 数据整理器: 若使用打包数据(已定长)则不需要动态padding
    def _maybe_resize_position_embeddings(model, new_len: int):
        old_len = model.config.max_position_embeddings
        if new_len <= old_len:
            return
        print(f"[警告] 扩展位置嵌入 {old_len} -> {new_len}. 这可能影响模型性能 (未预训练的区域)。")
        # 仅针对 BERT 结构
        emb = model.bert.embeddings.position_embeddings
        old_weight = emb.weight.data
        hidden_size = old_weight.size(1)
        new_emb = torch.nn.Embedding(new_len, hidden_size)
        # 拷贝已有
        new_emb.weight.data[:old_len] = old_weight
        # 用最后一个位置向量填充新增位置
        for pos in range(old_len, new_len):
            new_emb.weight.data[pos] = old_weight[-1]
        model.bert.embeddings.position_embeddings = new_emb
        model.config.max_position_embeddings = new_len

    if train_packed and eval_packed:
        packed_seq_len = train_dataset.input_ids.size(1)
        if packed_seq_len > model_max_pos:
            _maybe_resize_position_embeddings(model, packed_seq_len)
            effective_seq_len = packed_seq_len
        data_collator = None
        print(f"使用打包张量数据集: 序列定长 {packed_seq_len} (有效长度={effective_seq_len}) 跳过动态 padding")
    else:
        if effective_seq_len < model_max_pos:
            print(f"[信息] 将训练序列截断到 {effective_seq_len} <= 模型最大 {model_max_pos}")
        data_collator = PreTokenizedCollator(pad_token_id=tokenizer.pad_token_id or 0, max_seq_length=effective_seq_len, truncate=True)

    # 定义评估指标
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # 计算准确率
        accuracy = np.mean(predictions == labels)

        # 计算精确率、召回率和F1分数
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )

        # 计算混淆矩阵，并确保其为2x2
        cm = confusion_matrix(labels, predictions, labels=[0, 1])

        # 安全地解包混淆矩阵
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # 如果矩阵不是2x2（例如，只有一个类别出现），则手动设置值
            if len(np.unique(labels)) == 1:
                if np.unique(labels)[0] == 1:  # 只有正例
                    tp = cm[0][0] if cm.size == 1 else 0
                    tn, fp, fn = 0, 0, 0
                else:  # 只有负例
                    tn = cm[0][0] if cm.size == 1 else 0
                    tp, fp, fn = 0, 0, 0
            else:  # 理论上不应该发生，但作为保险
                tn, fp, fn, tp = 0, 0, 0, 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

    # 准备回调函数
    callbacks: list[TrainerCallback] = [
        TensorBoardCallback(tensorboard_log_dir)
    ]  # 显式类型注解

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
    # 检查tensorboard_log_dir是否存在来决定是否打印信息
    if "tensorboard_log_dir" in locals() and tensorboard_log_dir:
        print(f"\n=== TensorBoard 可视化 ===")
        print(f"TensorBoard 日志保存在: {tensorboard_log_dir}")
        print(f"启动 TensorBoard 查看训练过程:")
        print(f"  tensorboard --logdir={tensorboard_log_dir}")
        print(f"然后在浏览器中访问: http://localhost:6006")
        print(f"=========================\n")


if __name__ == "__main__":
    main()
