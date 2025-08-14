import os
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int
    support: int

    def to_dict(self):
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'true_positives': self.tp,
            'true_negatives': self.tn,
            'false_positives': self.fp,
            'false_negatives': self.fn,
            'support': self.support,
        }


def load_tokenized_file(path: str, label_key: str = 'label') -> Dict[str, List[Any]]:
    """读取已分词数据，支持 .jsonl (逐行) 与 .json (列表) 两种格式。"""
    input_ids_list: List[List[int]] = []
    attention_list: List[List[int]] = []
    labels: List[int] = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if 'input_ids' not in obj:
                    raise ValueError('already-tokenized 模式需要 input_ids 字段, 行缺失')
                input_ids_list.append(obj['input_ids'])
                attention_list.append(obj.get('attention_mask') or [1]*len(obj['input_ids']))
                lab_val = obj.get(label_key) if label_key in obj else obj.get('labels')
                if isinstance(lab_val, bool):
                    lab_val = 1 if lab_val else 0
                if lab_val is None:
                    lab_val = 0
                labels.append(int(lab_val))
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            root = json.load(f)
        if not isinstance(root, list):
            raise ValueError('tokenized json 顶层必须是列表')
        for obj in root:
            if not isinstance(obj, dict) or 'input_ids' not in obj:
                raise ValueError('tokenized json 列表元素需包含 input_ids 字段')
            input_ids_list.append(obj['input_ids'])
            attention_list.append(obj.get('attention_mask') or [1]*len(obj['input_ids']))
            lab_val = obj.get(label_key) if label_key in obj else obj.get('labels')
            if isinstance(lab_val, bool):
                lab_val = 1 if lab_val else 0
            if lab_val is None:
                lab_val = 0
            labels.append(int(lab_val))
    else:
        raise ValueError('只支持 .jsonl 或 .json 输入')
    return {'input_ids': input_ids_list, 'attention_mask': attention_list, 'labels': labels}


def load_text_file(path: str, text_key: str = 'text', label_key: str = 'label') -> Dict[str, List[Any]]:
    """读取原始文本数据 (.jsonl 或 .json 列表)。适配 preprocess 输出的 data_origin.json。"""
    texts: List[str] = []
    labels: List[int] = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError('JSONL 每行需要 object')
                text = obj.get(text_key)
                if not text:
                    continue
                lab_val = obj.get(label_key)
                if lab_val is None:
                    raise ValueError('缺少 label 字段')
                if isinstance(lab_val, bool):
                    lab_val = 1 if lab_val else 0
                texts.append(text)
                labels.append(int(lab_val))
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            root = json.load(f)
        if not isinstance(root, list):
            raise ValueError('json 顶层需为列表')
        for obj in root:
            if not isinstance(obj, dict):
                continue
            text = obj.get(text_key)
            if not text:
                continue
            lab_val = obj.get(label_key)
            if lab_val is None:
                continue  # 跳过缺 label 记录
            if isinstance(lab_val, bool):
                lab_val = 1 if lab_val else 0
            texts.append(text)
            labels.append(int(lab_val))
    else:
        raise ValueError('只支持 .jsonl 或 .json 输入')
    return {'texts': texts, 'labels': labels}


def batch_iter(seq_len: int, batch_size: int):
    for i in range(0, seq_len, batch_size):
        yield i, min(i+batch_size, seq_len)


def gpu_env_diag():
    print("=== 推理环境诊断 ===")
    print(f"torch: {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  {props.total_memory/1024**3:.1f} GB")
    print("====================")


def evaluate(model_dir: str,
             input: str,
             batch_size: int,
             max_length: int,
             already_tokenized: bool,
             text_key: str,
             label_key: str,
             save_predictions: str | None = None,
             ) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(model_dir, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    input_ids_list: List[List[int]] = []  # 为类型检查预先定义
    attn_list: List[List[int]] = []
    if already_tokenized:
        bundle = load_tokenized_file(input, label_key=label_key)
        input_ids_list = bundle['input_ids']
        attn_list = bundle['attention_mask']
        labels = bundle['labels']
        texts = None
    else:
        data = load_text_file(input, text_key=text_key, label_key=label_key)
        texts = data['texts']
        labels = data['labels']

    n = len(labels)
    print(f'[加载] 样本数: {n}  已分词: {already_tokenized}')

    preds: List[int] = []
    prob_1: List[float] = []
    id2label = model.config.id2label or {0: 'false', 1: 'true'}

    for start, end in tqdm(list(batch_iter(n, batch_size)), desc='推理'):
        if already_tokenized:
            batch_ids = input_ids_list[start:end]
            batch_attn = attn_list[start:end]
            max_len_batch = max(len(x) for x in batch_ids)
            if max_len_batch > max_length:
                max_len_batch = max_length
            input_ids_tensor = torch.full((end-start, max_len_batch), tokenizer.pad_token_id or 0, dtype=torch.long)
            attn_tensor = torch.zeros((end-start, max_len_batch), dtype=torch.long)
            for i,(ids,attn) in enumerate(zip(batch_ids, batch_attn)):
                ids = ids[:max_len_batch]
                attn = attn[:max_len_batch]
                input_ids_tensor[i,:len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn_tensor[i,:len(attn)] = torch.tensor(attn, dtype=torch.long)
            inputs = {
                'input_ids': input_ids_tensor.to(device),
                'attention_mask': attn_tensor.to(device),
            }
        else:
            batch_texts = texts[start:end]  # type: ignore
            inputs = tokenizer(batch_texts, truncation=True, max_length=max_length, padding=True, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            cls = torch.argmax(probs, dim=-1)
        preds.extend(cls.cpu().tolist())
        prob_1.extend(probs[:,1].cpu().tolist())

    # Metrics
    import numpy as np
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    accuracy = float((labels_arr == preds_arr).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(labels_arr, preds_arr, average='binary', zero_division=0)
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0,1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn=fp=fn=tp=0
    eval_res = EvalResult(accuracy, float(precision), float(recall), float(f1), int(tp), int(tn), int(fp), int(fn), int(len(labels)))

    report = classification_report(labels_arr, preds_arr, target_names=[id2label.get(0,'0'), id2label.get(1,'1')], zero_division=0)

    # 确保所有输出值为原生 Python 类型 (避免 numpy.int64/json 序列化报错)
    metrics_dict = eval_res.to_dict()
    metrics_dict = {k: (float(v) if isinstance(v, (float,)) else int(v) if isinstance(v, (bool, int)) else v) for k, v in metrics_dict.items()}
    out = {
        'metrics': metrics_dict,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'classification_report': report,
        'samples': int(n),
        'model': str(model_dir),
        'input': str(input),
        'already_tokenized': bool(already_tokenized),
        'max_length': int(max_length),
    }

    if save_predictions:
        with open(save_predictions, 'w', encoding='utf-8') as f:
            for i,(pred, prob) in enumerate(zip(preds, prob_1)):
                rec = {
                    'index': i,
                    'pred': int(pred),
                    'prob_valid': float(prob),
                    'label': int(labels[i])
                }
                if not already_tokenized and texts is not None:
                    rec['text'] = texts[i]
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'[输出] 预测写入 {save_predictions}')
    return out


def main():
    parser = argparse.ArgumentParser(description='评估脚本: 支持 jsonl (逐行) 与 json 列表 (data_origin.json)')
    parser.add_argument('--model', required=True, help='模型目录 (含 config.json, tokenizer)')
    parser.add_argument('--input', required=True, help='标注数据 (.jsonl 或 .json 列表)')
    parser.add_argument('--already-tokenized', action='store_true', help='输入为已分词 (含 input_ids/attention_mask) 的 jsonl/json')
    parser.add_argument('--text-key', default='text')
    parser.add_argument('--label-key', default='valid')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--save-predictions', default=None, help='保存逐样本预测 jsonl')
    parser.add_argument('--metrics-output', default=None, help='保存指标 json')
    parser.add_argument('--gpus', default=None, help='指定可见 GPU (如 "0" 或 "0,1")')
    args = parser.parse_args()

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f"设置 CUDA_VISIBLE_DEVICES={args.gpus}")
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    gpu_env_diag()

    res = evaluate(
        model_dir=args.model,
        input=args.input,
        batch_size=args.batch_size,
        max_length=args.max_length,
        already_tokenized=args.already_tokenized,
        text_key=args.text_key,
        label_key=args.label_key,
        save_predictions=args.save_predictions,
    )

    metrics_json = json.dumps(res, ensure_ascii=False, indent=2)
    if args.metrics_output:
        with open(args.metrics_output, 'w', encoding='utf-8') as f:
            f.write(metrics_json)
        print(f'[输出] 指标写入 {args.metrics_output}')
    else:
        print(metrics_json)


if __name__ == '__main__':
    main()
