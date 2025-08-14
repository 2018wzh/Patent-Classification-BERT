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
    """读取预先分词(tokenized)好的数据文件，支持以下格式，并提供 memory / stream 策略与自动回退。

    支持的输入格式:
      1) .jsonl 逐行 JSON（每行一个样本，典型的 tokenized_data.jsonl / train.jsonl 等）
      2) .json  列表 JSON（例如 preprocess 输出的 data_origin.json，可能尚未分词）

    若输入为未分词的 .json（列表），且对象中不存在 input_ids / attention_mask 字段，将尝试进行"即时分词"。
    即时分词要求调用方在实例化前已加载或可提供 tokenizer；为保持最小侵入，本类支持传入 tokenizer 与 max_seq_length。

    strategy (仅针对 .jsonl):
      - memory: 读取全部行到内存后解析 (速度快，需足够内存)
      - stream: 两遍文件扫描 (第一遍统计非空行数，第二遍解析)；内存占用低
    自动回退: memory 模式若发生 OSError 或 MemoryError，则回退到 stream。
    """

    def __init__(self,
                 path: str,
                 show_progress: bool = True,
                 strategy: str = "memory",
                 tokenizer: BertTokenizer | None = None,
                 text_column: str = "text",
                 label_column: str = "label",
                 max_seq_length: int | None = None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到数据文件: {path}")
        self.path = path
        self.samples: List[Dict[str, Any]] = []
        self._from_raw_text = False  # 标记是否进行了即时分词

        ext = os.path.splitext(path)[1].lower()
        if ext == '.jsonl':
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
        elif ext == '.json':
            # 读取整体 JSON 列表
            size_mb = os.path.getsize(self.path) / 1024 / 1024
            print(f"[Dataset] 加载 JSON 列表 {self.path} ({size_mb:.2f} MB)")
            with open(self.path, 'r', encoding='utf-8') as f:
                root = json.load(f)
            if not isinstance(root, list):
                raise ValueError(f"JSON 文件 {self.path} 顶层不是列表，无法解析为样本集合。")
            print(f"[Dataset] JSON 列表样本数: {len(root)}")
            # 判断是否已分词
            first = root[0] if root else {}
            already_tokenized = all(k in first for k in ("input_ids", "attention_mask", label_column))
            if not already_tokenized:
                if tokenizer is None:
                    raise ValueError("读取原始 JSON (未分词) 需要提供 tokenizer 参数。")
                if max_seq_length is None:
                    max_seq_length = 512
                self._from_raw_text = True
                texts: List[str] = []
                labels: List[int] = []
                for obj in root:
                    text_val = obj.get(text_column, "")
                    label_val = obj.get(label_column)
                    if isinstance(label_val, bool):
                        label_int = 1 if label_val else 0
                    elif isinstance(label_val, int):
                        label_int = int(label_val)
                    else:
                        # 对非 bool/int 的标签，若可解析为真值则置 1，否则 0
                        label_int = 1 if label_val else 0
                    texts.append(text_val)
                    labels.append(label_int)
                print(f"[Dataset] 即时分词 {len(texts)} 条样本 (max_seq_length={max_seq_length})")
                batch_size = 256
                for start in tqdm(range(0, len(texts), batch_size), desc="即时分词", unit="batch"):
                    end = min(start + batch_size, len(texts))
                    enc_batch = tokenizer(
                        texts[start:end],
                        padding=False,
                        max_length=max_seq_length,
                        truncation=True,
                        add_special_tokens=True,
                        return_attention_mask=True,
                    )
                    input_ids_list = list(enc_batch["input_ids"])  # type: ignore
                    attn_list = list(enc_batch["attention_mask"])  # type: ignore
                    for i, input_ids in enumerate(input_ids_list):
                        self.samples.append({
                            'input_ids': input_ids,
                            'attention_mask': attn_list[i],
                            'labels': labels[start + i],
                        })
                print(f"[Dataset] 即时分词完成，样本数={len(self.samples)}")
            else:
                for obj in root:
                    self.samples.append(self._normalize(obj, label_key=label_column))
        else:
            raise ValueError(f"不支持的文件扩展名: {ext} (仅支持 .jsonl / .json)")

    # ====== 内部方法 ======
    def _normalize(self, obj: Dict[str, Any], label_key: str | None = None) -> Dict[str, Any]:
        # 兼容不同字段名：优先使用显式 label_key，其次 fallback 'label'/'labels'
        if label_key and label_key in obj:
            label_val = obj.get(label_key)
        else:
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若输出目录存在 checkpoint-* 子目录，则自动从最新断点继续训练 (覆盖 config 中 resume_from_checkpoint=auto)",
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
    # 将 CLI resume 合并到 args; 若 config 中提供 resume_from_checkpoint 也保存
    config_resume = train_cfg_dict.get('resume_from_checkpoint')  # 可能是具体路径, 'auto', True/False
    setattr(args, 'resume_flag', bool(cli_args.resume) or (config_resume is True) or (config_resume == 'auto'))
    setattr(args, 'resume_from_checkpoint', config_resume if isinstance(config_resume, str) and config_resume not in {'auto'} else None)
    print(f"加载配置文件: {cli_args.config_file}")
    print(f"trainConfig 字段: {list(train_cfg_dict.keys())}")

    # 设置可见的GPU
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"设置 CUDA_VISIBLE_DEVICES='{args.gpus}'")

    # 关闭 tokenizers 多线程以减少多进程冲突
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 打印环境诊断信息
    def _env_diag():
        print("=== 环境诊断 ===")
        try:
            import torch
            print(f"torch={torch.__version__} cuda.is_available={torch.cuda.is_available()}")
            if torch.cuda.is_available():
                try:
                    import torch.version as torch_version  # type: ignore
                    cuda_ver = getattr(torch_version, 'cuda', 'unknown')
                except Exception:
                    cuda_ver = 'unknown'
                cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'NA'
                print(f"编译支持的CUDA: {cuda_ver}  cuDNN: {cudnn_ver}")
        except Exception as e:
            print(f"[env] 诊断失败: {e}")
        print(f"OS={os.name}  PID={os.getpid()}")
        print("==============")
    _env_diag()

    # 自动检测设备并打印信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f"可用GPU数量: {num_gpus}")
        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - 显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
            )
        if torch.cuda.device_count() > 1 and os.name == 'nt':
            print("[警告] Windows 多卡训练易出现底层 core dump (gloo/NCCL 限制)。建议使用: (1) WSL2 + Ubuntu + torchrun; 或 (2) 仅使用单卡；或 (3) 在 Linux 服务器上训练。")

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

    # 梯度检查点设置 (Windows 多卡时可能触发不稳定 / OOM, 先禁用)
    use_gradient_checkpointing = device.type == "cuda"
    if os.name == 'nt' and torch.cuda.device_count() > 1:
        use_gradient_checkpointing = False

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
        # 其余情况：可能是 .jsonl 或 .json (原始 / 已分词)
        try:
            ds = JsonlClassificationDataset(
                path,
                tokenizer=tokenizer,
                text_column=getattr(args, 'text_column_name', 'text') if hasattr(args, 'text_column_name') else 'text',
                label_column=getattr(args, 'label_column_name', 'label') if hasattr(args, 'label_column_name') else 'label',
                max_seq_length=effective_seq_len,
            )
            print(f"[{split}] 加载数据 ({path}) size={len(ds)}")
            return ds, False
        except Exception as e:
            raise RuntimeError(f"加载数据集失败 {path}: {e}")

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
        # 只有 PackedTensorDataset 才有 input_ids attribute
        if hasattr(train_dataset, 'input_ids'):
            packed_seq_len = train_dataset.input_ids.size(1)  # type: ignore
            if packed_seq_len > model_max_pos:
                _maybe_resize_position_embeddings(model, packed_seq_len)
                effective_seq_len = packed_seq_len
            print(f"使用打包张量数据集: 序列定长 {packed_seq_len} (有效长度={effective_seq_len}) 跳过动态 padding")
        else:
            print("[警告] 期望打包数据集具有 input_ids 张量, 但未找到, 仍继续训练。")
        data_collator = None
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

    # 5. 训练 & 评估 - 加保护捕获潜在 core dump 前的 Python 异常线索
    def _discover_latest_ckpt(out_dir: str):
        if not os.path.isdir(out_dir):
            return None
        subdirs = [d for d in os.listdir(out_dir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(out_dir,d))]
        if not subdirs:
            return None
        def _step(d):
            try:
                return int(d.split('-')[-1])
            except Exception:
                return -1
        subdirs.sort(key=_step, reverse=True)
        latest = subdirs[0]
        return os.path.join(out_dir, latest)

    resume_path = None
    if getattr(args, 'resume_from_checkpoint', None):
        if os.path.exists(args.resume_from_checkpoint):
            resume_path = args.resume_from_checkpoint
        else:
            print(f"[resume] 指定的 resume_from_checkpoint 路径不存在: {args.resume_from_checkpoint}")
    elif getattr(args, 'resume_flag', False):
        # 自动发现
        resume_path = _discover_latest_ckpt(args.output_dir)
        if resume_path:
            print(f"[resume] 自动发现最新断点: {resume_path}")
        else:
            print("[resume] 未发现 checkpoint-* 目录，将从头训练。")
    else:
        # 若未显式要求但 output_dir 中存在 checkpoint, 给出提示 (防止误覆盖)
        potential = _discover_latest_ckpt(args.output_dir)
        if potential:
            print(f"[提示] 发现已有断点 {potential}. 如需续训请添加 --resume 或在 config 中设置 'resume_from_checkpoint': 'auto'")
    try:
        train_result = trainer.train()
        if resume_path:
            print(f"[训练] 从断点继续: {resume_path}")
            train_result = trainer.train(resume_from_checkpoint=resume_path)
        else:
            train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    except RuntimeError as e:
        print(f"[训练RuntimeError] {e}")
        print("排查建议: \n"
              "1. 降低 per_device_train_batch_size (例如减半)。\n"
              "2. 设置环境变量 TORCH_USE_CUDA_DSA=1 以捕获越界 (Linux).\n"
              "3. 关闭梯度检查点 use_gradient_checkpointing=False。\n"
              "4. 若是 CUDNN 状态错误, 设置 torch.backends.cudnn.deterministic=True。\n"
              "5. 使用单卡验证是否稳定, 再扩展多卡。\n"
              "6. 在 Linux/WSL2 下使用: torchrun --nproc_per_node=<gpus> train.py --config_file config/config.json")
        raise

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
