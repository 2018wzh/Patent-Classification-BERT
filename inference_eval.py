import os
import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

# 复用 preprocess 中的逻辑 (CSV -> 结构化 + 主分类号匹配)
try:
    from preprocess import (
        load_config as pp_load_config,
        process_csv_file as pp_process_csv_file,
        normalize_single_ipc as pp_normalize_single_ipc,
    )
except Exception:
    pp_load_config = None  # type: ignore
    pp_process_csv_file = None  # type: ignore
    pp_normalize_single_ipc = None  # type: ignore
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


def load_tokenized_file(path: str, label_key: str = 'label', ipc_key: str = 'ipc') -> Dict[str, List[Any]]:
    """读取已分词数据，支持 .jsonl (逐行) 与 .json (列表) 两种格式。"""
    input_ids_list: List[List[int]] = []
    attention_list: List[List[int]] = []
    labels: List[int] = []
    ipc: List[str] = []
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
                ipc.append(obj.get(ipc_key, ''))
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
            ipc.append(obj.get(ipc_key, ''))
    else:
        raise ValueError('只支持 .jsonl 或 .json 输入')
    return {'input_ids': input_ids_list, 'attention_mask': attention_list, 'labels': labels, 'ipc': ipc}


def load_text_file(path: str, text_key: str = 'text', label_key: str = 'label', ipc_key: str = 'ipc') -> Dict[str, List[Any]]:
    """读取原始文本数据 (.jsonl 或 .json 列表)。适配 preprocess 输出的 data_origin.json。"""
    texts: List[str] = []
    labels: List[int] = []
    ipc: List[str] = []
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
                ipc.append(obj.get(ipc_key, ''))
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
            ipc.append(obj.get(ipc_key, ''))
    else:
        raise ValueError('只支持 .jsonl 或 .json 输入')
    return {'texts': texts, 'labels': labels, 'ipc': ipc}


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
             input: Optional[str],
             batch_size: int,
             max_length: int,
             already_tokenized: bool,
             text_key: str,
             label_key: str,
             save_predictions: Optional[str] = None,
             in_memory_texts: Optional[List[str]] = None,
             in_memory_labels: Optional[List[int]] = None,
             ipc_key: str = 'ipc',
             in_memory_ipcs: Optional[List[str]] = None,
             ) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(model_dir, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    # 赋值回 model 以兼容部分类型检查器
    model = model.to(device)  # type: ignore
    model.eval()

    input_ids_list: List[List[int]] = []  # 为类型检查预先定义
    attn_list: List[List[int]] = []
    ipc: List[str] | None = None
    if in_memory_texts is not None and in_memory_labels is not None:
        # 内存模式 (例如来自 CSV 直接预处理)
        if already_tokenized:
            raise ValueError('内存模式暂不支持 already_tokenized=True')
        texts = in_memory_texts
        labels = in_memory_labels
        if in_memory_ipcs is not None:
            ipc = in_memory_ipcs
    else:
        if input is None:
            raise ValueError('未提供 input 路径或内存数据')
        if already_tokenized:
            bundle = load_tokenized_file(input, label_key=label_key, ipc_key=ipc_key)
            input_ids_list = bundle['input_ids']
            attn_list = bundle['attention_mask']
            labels = bundle['labels']
            ipc = bundle.get('ipc')
            texts = None
        else:
            data = load_text_file(input, text_key=text_key, label_key=label_key, ipc_key=ipc_key)
            texts = data['texts']
            labels = data['labels']
            ipc = data.get('ipc')

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
    # IPC 关联统计
    ipc_stats: List[Dict[str, Any]] = []
    if ipc is not None and len(ipc) == n:
        agg: Dict[str, Dict[str, int]] = {}
        for i, ipc_val in enumerate(ipc):
            key = ipc_val or ''
            a = agg.get(key)
            if a is None:
                a = {'total':0,'label_pos':0,'pred_pos':0,'correct':0,'tp_pos':0}
                agg[key] = a
            a['total'] += 1
            lab = labels[i]
            pred = preds[i]
            if lab == 1:
                a['label_pos'] += 1
            if pred == 1:
                a['pred_pos'] += 1
            if pred == lab:
                a['correct'] += 1
            if pred == 1 and lab == 1:
                a['tp_pos'] += 1
        # 计算派生指标
        for k, a in agg.items():
            total_k = a['total']
            prec = a['tp_pos']/a['pred_pos'] if a['pred_pos'] else 0.0
            rec = a['tp_pos']/a['label_pos'] if a['label_pos'] else 0.0
            f1_k = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
            ipc_stats.append({
                'ipc': k,
                'total': total_k,
                'accuracy': a['correct']/total_k if total_k else 0.0,
                'label_pos': a['label_pos'],
                'pred_pos': a['pred_pos'],
                'tp_pos': a['tp_pos'],
                'precision_pos': prec,
                'recall_pos': rec,
                'f1_pos': f1_k,
            })
        ipc_stats.sort(key=lambda x: x['total'], reverse=True)

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
        'input': str(input) if input else 'in-memory',
        'already_tokenized': bool(already_tokenized),
        'max_length': int(max_length),
        'ipc_stats': ipc_stats,
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
                if ipc is not None and len(ipc)==n:
                    rec['ipc'] = ipc[i]
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'[输出] 预测写入 {save_predictions}')
    return out


def main():
    parser = argparse.ArgumentParser(description='评估脚本: 支持 jsonl/json 以及 csv (直接预处理内存推理)')
    parser.add_argument('--model', required=True, help='模型目录 (含 config.json, tokenizer)')
    parser.add_argument('--input', required=True, help='标注数据 (.jsonl/.json 或 .csv)')
    parser.add_argument('--already-tokenized', action='store_true', help='输入为已分词 (含 input_ids/attention_mask) 的 jsonl/json')
    parser.add_argument('--text-key', default='text')
    parser.add_argument('--label-key', default='label')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--save-predictions', default=None, help='保存逐样本预测 jsonl')
    parser.add_argument('--metrics-output', default=None, help='保存指标 json')
    parser.add_argument('--gpus', default=None, help='指定可见 GPU (如 "0" 或 "0,1")')
    parser.add_argument('--config', default='config/config.json', help='当输入为 CSV 时加载的配置文件 (含 preprocessConfig.validLabels/removeKeywords)')
    parser.add_argument('--ipc-key', default='ipc', help='IPC 字段名称 (用于统计)')
    parser.add_argument('--ipc-top-k', type=int, default=100, help='IPC 汇总中前K名 (比例/数量)')
    parser.add_argument('--ipc-min-total', type=int, default=1, help='计算最高占比/前K占比列表时的最小样本数过滤 (默认不过滤)')
    parser.add_argument('--ipc-summary-text', default=None, help='可选: 将 IPC 汇总指标写出为纯文本文件 (类似示例)')
    args = parser.parse_args()

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f"设置 CUDA_VISIBLE_DEVICES={args.gpus}")
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    gpu_env_diag()

    # CSV 模式: 读取 + 直接调用 preprocess 逻辑于内存中完成匹配
    if args.input.lower().endswith('.csv'):
        if pp_load_config is None or pp_process_csv_file is None or pp_normalize_single_ipc is None:
            raise RuntimeError('无法导入 preprocess 中的函数，请确保 preprocess.py 存在且可导入')
        if not os.path.exists(args.config):
            raise FileNotFoundError(f'配置文件不存在: {args.config}')
        cfg_all = pp_load_config(args.config)
        if 'preprocessConfig' not in cfg_all:
            raise ValueError('配置文件缺少 preprocessConfig 节点')
        pp_cfg = cfg_all['preprocessConfig']
        remove_keywords = pp_cfg.get('removeKeywords') or []
        valid_labels_cfg = pp_cfg.get('validLabels') or []
        print(f'[CSV] 读取并预处理: {args.input}')
        records = pp_process_csv_file(args.input, remove_keywords)
        print(f'[CSV] 原始记录数: {len(records)}  有效前缀候选: {len(valid_labels_cfg)}')
        # 规范化前缀集合 (长优先)
        valid_norm = sorted({pp_normalize_single_ipc(v) for v in valid_labels_cfg if v}, key=lambda x: (-len(x), x))
        matched = 0
        texts_mem: List[str] = []
        labels_mem: List[int] = []
        ipcs_mem: List[str] = []
        for r in records:
            ipc_code = pp_normalize_single_ipc(r.get('ipc',''))
            lab = 0
            for pref in valid_norm:
                if ipc_code.startswith(pref):
                    lab = 1
                    break
            if lab == 1:
                matched += 1
            texts_mem.append(r.get('text',''))
            labels_mem.append(lab)
            ipcs_mem.append(r.get('ipc',''))
        print(f'[CSV] 匹配成功(label=1): {matched}  未匹配(label=0): {len(records)-matched}')
        res = evaluate(
            model_dir=args.model,
            input=None,
            batch_size=args.batch_size,
            max_length=args.max_length,
            already_tokenized=False,
            text_key=args.text_key,
            label_key=args.label_key,
            save_predictions=args.save_predictions,
            in_memory_texts=texts_mem,
            in_memory_labels=labels_mem,
            ipc_key=args.ipc_key,
            in_memory_ipcs=ipcs_mem,
        )
    else:
        res = evaluate(
            model_dir=args.model,
            input=args.input,
            batch_size=args.batch_size,
            max_length=args.max_length,
            already_tokenized=args.already_tokenized,
            text_key=args.text_key,
            label_key=args.label_key,
            save_predictions=args.save_predictions,
            ipc_key=args.ipc_key,
        )

    # 基于已存在的 ipc_stats 生成汇总 (若存在)
    ipc_stats = res.get('ipc_stats') or []
    if ipc_stats:
        top_k = max(1, args.ipc_top_k)
        min_total = max(1, args.ipc_min_total)
        # 重新构建统计以确保有 pos/total 信息
        # ipc_stats 中已含: ipc,total,label_pos
        # 正例数=label_pos, 占比=label_pos/total
        enriched = []
        for row in ipc_stats:
            total = int(row.get('total',0))
            pos = int(row.get('label_pos',0))
            if total <= 0:
                continue
            ratio = pos/total if total else 0.0
            enriched.append((row.get('ipc',''), total, pos, ratio))
        distinct_count = len(enriched)
        total_samples = sum(t for _,t,_,_ in enriched)
        total_positive = sum(p for _,_,p,_ in enriched)
        overall_ratio = total_positive/total_samples if total_samples else 0.0
        # 过滤后用于比例 TopK
        filtered_for_ratio = [e for e in enriched if e[1] >= min_total]
        if filtered_for_ratio:
            max_ratio_entry = max(filtered_for_ratio, key=lambda x: (x[3], x[1]))
        else:
            max_ratio_entry = None
        # 比例排序
        top_ratio = sorted(filtered_for_ratio, key=lambda x: (-x[3], -x[1], x[0]))[:top_k]
        # 数量排序 (不必过滤)
        top_count = sorted(enriched, key=lambda x: (-x[1], -x[3], x[0]))[:top_k]
        summary = {
            'distinct_ipc': distinct_count,
            'total_samples': total_samples,
            'total_positive': total_positive,
            'overall_positive_ratio': overall_ratio,
            'max_positive_ratio_ipc': {
                'ipc': max_ratio_entry[0],
                'total': max_ratio_entry[1],
                'positive': max_ratio_entry[2],
                'positive_ratio': max_ratio_entry[3],
            } if max_ratio_entry else None,
            'top_positive_ratio': [
                {
                    'ipc': ipc,
                    'total': tot,
                    'positive': pos,
                    'positive_ratio': ratio,
                } for ipc,tot,pos,ratio in top_ratio
            ],
            'top_total': [
                {
                    'ipc': ipc,
                    'total': tot,
                    'positive': pos,
                    'positive_ratio': ratio,
                } for ipc,tot,pos,ratio in top_count
            ],
            'min_total_filter': min_total,
            'top_k': top_k,
        }
        # 找出样本最多 IPC (top_total 第一项)
        if top_count:
            summary['max_total_ipc'] = {
                'ipc': top_count[0][0],
                'total': top_count[0][1],
                'positive': top_count[0][2],
                'positive_ratio': top_count[0][3],
            }
        res['ipc_summary'] = summary

        if args.ipc_summary_text:
            try:
                lines = []
                lines.append(f"统计了 {summary['distinct_ipc']} 个不同的 IPC 代码")
                lines.append(f"总样本数: {summary['total_samples']}")
                lines.append(f"总正例数: {summary['total_positive']}")
                lines.append(f"总体正例占比: {summary['overall_positive_ratio']:.4f}")
                if summary.get('max_positive_ratio_ipc'):
                    m = summary['max_positive_ratio_ipc']
                    lines.append(f"最高正例占比的 IPC: {m['ipc']} ({m['positive_ratio']:.4f}) ({m['positive']}/{m['total']})")
                if summary.get('max_total_ipc'):
                    m2 = summary['max_total_ipc']
                    lines.append(f"样本最多的 IPC: {m2['ipc']} ({m2['total']} 条, 正例占比 {m2['positive_ratio']:.4f})")
                lines.append("")
                lines.append(f"正例占比前 {summary['top_k']} 名 (min_total={summary['min_total_filter']}):")
                for e in summary['top_positive_ratio']:
                    lines.append(f"  {e['ipc']}: {e['positive_ratio']:.4f} ({e['positive']}/{e['total']})")
                lines.append("")
                lines.append(f"样本数前 {summary['top_k']} 名:")
                for e in summary['top_total']:
                    lines.append(f"  {e['ipc']}: {e['total']} 条 (正例占比 {e['positive_ratio']:.4f})")
                with open(args.ipc_summary_text, 'w', encoding='utf-8') as ftxt:
                    ftxt.write('\n'.join(lines))
                print(f"[输出] IPC 文本汇总写入 {args.ipc_summary_text}")
            except Exception as e:
                print(f"[警告] 写入 IPC 文本汇总失败: {e}")

    metrics_json = json.dumps(res, ensure_ascii=False, indent=2)
    if args.metrics_output:
        with open(args.metrics_output, 'w', encoding='utf-8') as f:
            f.write(metrics_json)
        print(f'[输出] 指标写入 {args.metrics_output}')
    else:
        print(metrics_json)


if __name__ == '__main__':
    main()
