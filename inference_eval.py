import os
import glob
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.distributed as dist
import gc
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

# 复用 preprocess 中的逻辑 (CSV -> 结构化 + 主分类号匹配)
try:
    from preprocess import (
        load_config as pp_load_config,
        process_csv_file as pp_process_csv_file,
        normalize_single_ipc as pp_normalize_single_ipc,
        process_csv_file_stream as pp_process_csv_file_stream,
    )
except Exception:
    pp_load_config = None  # type: ignore
    pp_process_csv_file = None  # type: ignore
    pp_normalize_single_ipc = None  # type: ignore
    pp_process_csv_file_stream = None  # type: ignore
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
    ids: List[Any] = []
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
                ids.append(obj.get('id'))
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
            ids.append(obj.get('id'))
    else:
        raise ValueError('只支持 .jsonl 或 .json 输入')
    return {'input_ids': input_ids_list, 'attention_mask': attention_list, 'labels': labels, 'ipc': ipc, 'ids': ids}


def load_text_file(path: str, text_key: str = 'text', label_key: str = 'label', ipc_key: str = 'ipc') -> Dict[str, List[Any]]:
    """读取原始文本数据 (.jsonl 或 .json 列表)。适配 preprocess 输出的 data_origin.json。"""
    texts: List[str] = []
    labels: List[int] = []
    ipc: List[str] = []
    ids: List[Any] = []
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
                ids.append(obj.get('id'))
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
            ids.append(obj.get('id'))
    else:
        raise ValueError('只支持 .jsonl 或 .json 输入')
    return {'texts': texts, 'labels': labels, 'ipc': ipc, 'ids': ids}


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


def _maybe_init_dist() -> Tuple[bool, int, int, int]:
    """返回 (is_dist, rank, world_size, local_rank)。如在 torchrun 环境则初始化进程组并设置设备。"""
    try:
        # torchrun 会设置以下环境变量
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('LOCAL_PROCESS_RANK', '0')))
        is_dist = world_size > 1
        if is_dist and not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, init_method='env://')
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
            except Exception:
                pass
        return is_dist, rank, world_size, local_rank
    except Exception:
        return False, 0, 1, 0


def _ensure_tok_model_compat(tokenizer: BertTokenizer, model: BertForSequenceClassification):
    """确保 tokenizer 词表大小与模型嵌入大小一致；若 pad_token 缺失则添加并调整嵌入。
    防止 input_ids 超出 embeddings.num_embeddings 导致 GPU gather 越界。
    """
    try:
        vocab_sz_tok = len(tokenizer)
        emb = model.get_input_embeddings()
        vocab_sz_model = int(getattr(emb, 'num_embeddings', model.config.vocab_size))
        # 确保 pad_token 存在
        if tokenizer.pad_token_id is None:
            pad_token = tokenizer.eos_token or tokenizer.sep_token or '[PAD]'
            tokenizer.add_special_tokens({'pad_token': pad_token})
            vocab_sz_tok = len(tokenizer)
        if vocab_sz_tok != vocab_sz_model:
            model.resize_token_embeddings(vocab_sz_tok)
    except Exception as e:
        print(f"[警告] 检查/对齐 tokenizer-模型 词表时出错: {e}")


def _cuda_cleanup():
    """尽力释放 CUDA 显存与 Python 引用。"""
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                # 一些平台支持 IPC 缓存回收
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def _autocast_ctx(device: torch.device, precision: str):
    """返回自动混合精度上下文。precision: 'fp32'|'fp16'|'bf16'|'auto'。CPU 下返回空上下文。"""
    try:
        if device.type != 'cuda':
            return nullcontext()
        if precision not in ('fp16', 'bf16', 'auto'):
            return nullcontext()
        # 优先使用 torch.autocast (新 API)
        try:
            from torch import autocast as torch_autocast  # type: ignore
            if precision == 'bf16':
                return torch_autocast(device_type='cuda', dtype=torch.bfloat16)
            return torch_autocast(device_type='cuda', dtype=torch.float16)
        except Exception:
            pass
        # 回退到 torch.cuda.amp.autocast (旧 API)
        try:
            from torch.cuda.amp import autocast as cuda_autocast  # type: ignore
            if precision == 'bf16':
                return cuda_autocast(dtype=torch.bfloat16)
            return cuda_autocast(dtype=torch.float16)
        except Exception:
            return nullcontext()
    except Exception:
        return nullcontext()


def _total_from_metadata(input_path: str) -> Optional[int]:
    """尝试从同目录 metadata.json 获取该输入对应的样本总数，用于进度条 total。"""
    try:
        meta_path = os.path.join(os.path.dirname(input_path), 'metadata.json')
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        split_paths = (meta.get('split_paths') or {})
        splits_info = (meta.get('splits') or {})
        # 优先匹配分割文件
        for name, p in split_paths.items():
            try:
                if os.path.abspath(p) == os.path.abspath(input_path):
                    v = splits_info.get(name)
                    if isinstance(v, (int, float)):
                        return int(v)
            except Exception:
                continue
        # 退回匹配 tokenized/origin
        paths = meta.get('paths') or {}
        counts = meta.get('counts') or {}
        for k in ('tokenized_jsonl', 'origin_jsonl'):
            p = paths.get(k)
            if p and os.path.abspath(p) == os.path.abspath(input_path):
                v = counts.get('total_records')
                if isinstance(v, (int, float)):
                    return int(v)
        # 进一步回退：若是 data_origin.jsonl 且与 metadata 同目录，则直接返回 total_records
        base = os.path.basename(input_path).lower()
        if base.startswith('data_origin') or base == 'origin.jsonl':
            v = (meta.get('counts') or {}).get('total_records')
            if isinstance(v, (int, float)) and v:
                return int(v)
        # 最后回退：同目录任意 jsonl，若 counts.total_records 存在，则作为估计值
        v_any = (meta.get('counts') or {}).get('total_records')
        if isinstance(v_any, (int, float)) and v_any:
            return int(v_any)
    except Exception:
        return None
    return None


def evaluate_stream(
    model_dir: str,
    input: str,
    batch_size: int,
    max_length: int,
    threshold: float,
    already_tokenized: bool,
    text_key: str,
    label_key: str,
    save_predictions: Optional[str] = None,
    ipc_key: str = 'ipc',
    shard_within_file: bool = False,
    rank: int = 0,
    world_size: int = 1,
    shared_writer: Optional[Any] = None,
    source_name: Optional[str] = None,
    index_offset: int = 0,
    amp_precision: str = 'auto',
    empty_cache_interval: int = 200,
    pad_to_max_length: bool = False,
    profile: bool = False,
) -> Dict[str, Any]:
    """流式评估：一边读取一边(可选分词)一边推理，并增量写出预测与累计指标。仅支持 .jsonl。"""
    if not input.lower().endswith('.jsonl'):
        raise ValueError('流式模式仅支持 JSONL 输入')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(model_dir, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)  # type: ignore
    _ensure_tok_model_compat(tokenizer, model)
    model.eval()

    total = _total_from_metadata(input)
    # 计数器
    tp=tn=fp=fn=0
    n_total = 0
    prob_1_all: List[float] = []  # 仅用于可选写出，不整体保留 preds/labels 以节省内存
    preds_tmp: List[int] = []
    labels_tmp: List[int] = []
    ipc_agg: Dict[str, Dict[str, int]] = {}
    local_opened = False
    writer = shared_writer
    if writer is None and save_predictions:
        writer = open(save_predictions, 'w', encoding='utf-8')
        local_opened = True

    def update_ipc(ipc_list: List[str], labels_b: List[int], preds_b: List[int]):
        if not ipc_list:
            return
        for i, k in enumerate(ipc_list):
            key = k or ''
            a = ipc_agg.get(key)
            if a is None:
                a = {'total':0,'label_pos':0,'pred_pos':0,'correct':0,'tp_pos':0}
                ipc_agg[key] = a
            a['total'] += 1
            lab = labels_b[i]
            pred = preds_b[i]
            if lab == 1:
                a['label_pos'] += 1
            if pred == 1:
                a['pred_pos'] += 1
            if pred == lab:
                a['correct'] += 1
            if pred == 1 and lab == 1:
                a['tp_pos'] += 1

    try:
        with open(input, 'r', encoding='utf-8') as f:
            pbar = tqdm(f, total=total, desc='推理(流式)', unit='行')
            # 批缓冲
            buf_ids: List[List[int]] = []
            buf_attn: List[List[int]] = []
            buf_texts: List[str] = []
            buf_labels: List[int] = []
            buf_ipc: List[str] = []
            buf_rec_ids: List[Any] = []
            idx = 0
            line_index = 0
            batch_counter = 0
            import time
            tok_t = fwd_t = io_t = 0.0
            for line in pbar:
                line = line.strip()
                if not line:
                    continue
                # 仅当需要在文件内分片时，按行号 % world_size 选取
                if shard_within_file and world_size > 1:
                    if (line_index % world_size) != rank:
                        line_index += 1
                        continue
                    line_index += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                lab_val = obj.get(label_key) if label_key in obj else obj.get('labels')
                if isinstance(lab_val, bool):
                    lab_val = 1 if lab_val else 0
                if lab_val is None:
                    # 跳过无标签
                    continue
                lab = int(lab_val)
                buf_labels.append(lab)
                buf_ipc.append(obj.get(ipc_key, ''))
                if already_tokenized:
                    ids_token = obj.get('input_ids') or []
                    attn = obj.get('attention_mask') or [1]*len(ids_token)
                    buf_ids.append(ids_token)
                    buf_attn.append(attn)
                else:
                    text = obj.get(text_key)
                    if not text:
                        # 无文本则跳过
                        buf_labels.pop(); buf_ipc.pop()
                        continue
                    buf_texts.append(text)
                # 记录该样本的原始 id（如存在）
                buf_rec_ids.append(obj.get('id'))

                if len(buf_labels) >= batch_size:
                    # 处理一个批
                    if already_tokenized:
                        max_len_batch = min(max_length, max((len(x) for x in buf_ids), default=1))
                        input_ids_tensor = torch.full((len(buf_ids), max_len_batch), tokenizer.pad_token_id or 0, dtype=torch.long)
                        attn_tensor = torch.zeros((len(buf_attn), max_len_batch), dtype=torch.long)
                        for i,(ids,attn) in enumerate(zip(buf_ids, buf_attn)):
                            ids = ids[:max_len_batch]
                            attn = attn[:max_len_batch]
                            input_ids_tensor[i,:len(ids)] = torch.tensor(ids, dtype=torch.long)
                            attn_tensor[i,:len(attn)] = torch.tensor(attn, dtype=torch.long)
                        inputs = { 'input_ids': input_ids_tensor.to(device), 'attention_mask': attn_tensor.to(device) }
                    else:
                        t0 = time.perf_counter()
                        enc = tokenizer(
                            buf_texts,
                            truncation=True,
                            max_length=max_length,
                            padding=('max_length' if pad_to_max_length else True),
                            return_tensors='pt'
                        )
                        tok_t += (time.perf_counter() - t0)
                        inputs = {k: v.to(device) for k,v in enc.items()}
                    with torch.inference_mode():
                        t1 = time.perf_counter()
                        with _autocast_ctx(device, amp_precision):
                            logits = model(**inputs).logits
                            probs = torch.softmax(logits, dim=-1)
                        fwd_t += (time.perf_counter() - t1)
                    # 阈值判定
                    prob1_b = probs[:,1].detach().cpu().tolist()
                    preds_b = [1 if p > threshold else 0 for p in prob1_b]
                    # 更新指标
                    for j, pred in enumerate(preds_b):
                        gt = buf_labels[j]
                        if pred == 1 and gt == 1:
                            tp += 1
                        elif pred == 0 and gt == 0:
                            tn += 1
                        elif pred == 1 and gt == 0:
                            fp += 1
                        else:
                            fn += 1
                    n_total += len(preds_b)
                    update_ipc(buf_ipc, buf_labels, preds_b)
                    if writer is not None:
                        lines = []
                        for j, pred in enumerate(preds_b):
                            rec = {
                                'index': index_offset + idx + j,
                                'pred': int(pred),
                                'prob_valid': float(prob1_b[j]),
                                'label': int(buf_labels[j])
                            }
                            if not already_tokenized:
                                rec['text'] = buf_texts[j]
                            if buf_ipc:
                                rec['ipc'] = buf_ipc[j]
                            if source_name:
                                rec['source'] = source_name
                            if len(buf_rec_ids) > j and buf_rec_ids[j] is not None:
                                rec['id'] = buf_rec_ids[j]
                            lines.append(json.dumps(rec, ensure_ascii=False) + '\n')
                        t2 = time.perf_counter(); writer.write(''.join(lines)); io_t += (time.perf_counter() - t2)
                    idx += len(preds_b)
                    # 清空批
                    buf_ids.clear(); buf_attn.clear(); buf_texts.clear(); buf_labels.clear(); buf_ipc.clear(); buf_rec_ids.clear()
                    batch_counter += 1
                    if torch.cuda.is_available() and empty_cache_interval > 0 and (batch_counter % empty_cache_interval == 0):
                        # 周期性释放缓存显存，避免峰值堆积
                        torch.cuda.empty_cache()

            # 尾批
            if buf_labels:
                if already_tokenized:
                    max_len_batch = min(max_length, max((len(x) for x in buf_ids), default=1))
                    input_ids_tensor = torch.full((len(buf_ids), max_len_batch), tokenizer.pad_token_id or 0, dtype=torch.long)
                    attn_tensor = torch.zeros((len(buf_attn), max_len_batch), dtype=torch.long)
                    for i,(ids,attn) in enumerate(zip(buf_ids, buf_attn)):
                        ids = ids[:max_len_batch]
                        attn = attn[:max_len_batch]
                        input_ids_tensor[i,:len(ids)] = torch.tensor(ids, dtype=torch.long)
                        attn_tensor[i,:len(attn)] = torch.tensor(attn, dtype=torch.long)
                    inputs = { 'input_ids': input_ids_tensor.to(device), 'attention_mask': attn_tensor.to(device) }
                else:
                    t0 = time.perf_counter()
                    enc = tokenizer(
                        buf_texts,
                        truncation=True,
                        max_length=max_length,
                        padding=('max_length' if pad_to_max_length else True),
                        return_tensors='pt'
                    )
                    tok_t += (time.perf_counter() - t0)
                    inputs = {k: v.to(device) for k,v in enc.items()}
                with torch.inference_mode():
                    t1 = time.perf_counter()
                    with _autocast_ctx(device, amp_precision):
                        logits = model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1)
                    fwd_t += (time.perf_counter() - t1)
                prob1_b = probs[:,1].detach().cpu().tolist()
                preds_b = [1 if p > threshold else 0 for p in prob1_b]
                for j, pred in enumerate(preds_b):
                    gt = buf_labels[j]
                    if pred == 1 and gt == 1:
                        tp += 1
                    elif pred == 0 and gt == 0:
                        tn += 1
                    elif pred == 1 and gt == 0:
                        fp += 1
                    else:
                        fn += 1
                n_total += len(preds_b)
                update_ipc(buf_ipc, buf_labels, preds_b)
                if writer is not None:
                    lines = []
                    for j, pred in enumerate(preds_b):
                        rec = {
                            'index': index_offset + idx + j,
                            'pred': int(pred),
                            'prob_valid': float(prob1_b[j]),
                            'label': int(buf_labels[j])
                        }
                        if not already_tokenized:
                            rec['text'] = buf_texts[j]
                        if buf_ipc:
                            rec['ipc'] = buf_ipc[j]
                        if source_name:
                            rec['source'] = source_name
                        if len(buf_rec_ids) > j and buf_rec_ids[j] is not None:
                            rec['id'] = buf_rec_ids[j]
                        lines.append(json.dumps(rec, ensure_ascii=False) + '\n')
                    t2 = time.perf_counter(); writer.write(''.join(lines)); io_t += (time.perf_counter() - t2)
                if torch.cuda.is_available() and empty_cache_interval > 0:
                    torch.cuda.empty_cache()
                # 无需清空，循环结束
            if profile:
                total = tok_t + fwd_t + io_t
                if total > 0:
                    print(f"[性能] tokenization {tok_t:.3f}s ({tok_t/total:.1%}), forward {fwd_t:.3f}s ({fwd_t/total:.1%}), io {io_t:.3f}s ({io_t/total:.1%})")
    except KeyboardInterrupt:
        if profile:
            total = tok_t + fwd_t + io_t
            if total > 0:
                print(f"[性能] tokenization {tok_t:.3f}s ({tok_t/total:.1%}), forward {fwd_t:.3f}s ({fwd_t/total:.1%}), io {io_t:.3f}s ({io_t/total:.1%}) （中断，部分统计）")
    finally:
        if local_opened and writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        _cuda_cleanup()
    # 汇总指标
    accuracy = (tp + tn) / max(1, (tp+tn+fp+fn))
    precision = tp / max(1, (tp+fp))
    recall = tp / max(1, (tp+fn))
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    eval_res = EvalResult(accuracy, precision, recall, f1, tp, tn, fp, fn, n_total)

    ipc_stats: List[Dict[str, Any]] = []
    for k, a in ipc_agg.items():
        total_k = a['total']
        prec_k = a['tp_pos']/a['pred_pos'] if a['pred_pos'] else 0.0
        rec_k = a['tp_pos']/a['label_pos'] if a['label_pos'] else 0.0
        f1_k = (2*prec_k*rec_k/(prec_k+rec_k)) if (prec_k+rec_k)>0 else 0.0
        ipc_stats.append({
            'ipc': k,
            'total': total_k,
            'accuracy': a['correct']/total_k if total_k else 0.0,
            'label_pos': a['label_pos'],
            'pred_pos': a['pred_pos'],
            'tp_pos': a['tp_pos'],
            'precision_pos': prec_k,
            'recall_pos': rec_k,
            'f1_pos': f1_k,
        })
    ipc_stats.sort(key=lambda x: x['total'], reverse=True)

    report_text = (
        f"Streaming report (binary)\n"
        f"samples={n_total} accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}\n"
        f"tn={tn} fp={fp} fn={fn} tp={tp}\n"
    )

    out = {
        'metrics': eval_res.to_dict(),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'classification_report': report_text,
        'samples': int(n_total),
        'model': str(model_dir),
        'input': str(input),
        'already_tokenized': bool(already_tokenized),
        'max_length': int(max_length),
    'threshold': float(threshold),
        'ipc_stats': ipc_stats,
    'predictions_written': int(n_total),
    }
    return out


def evaluate_stream_csv(
    model_dir: str,
    csv_path: str,
    batch_size: int,
    max_length: int,
    threshold: float,
    text_key: str,
    label_key: str,
    save_predictions: Optional[str] = None,
    ipc_key: str = 'ipc',
    remove_keywords: Optional[List[str]] = None,
    valid_labels_cfg: Optional[List[str]] = None,
    shard_within_file: bool = False,
    rank: int = 0,
    world_size: int = 1,
    shared_writer: Optional[Any] = None,
    source_name: Optional[str] = None,
    index_offset: int = 0,
    amp_precision: str = 'auto',
    empty_cache_interval: int = 200,
    pad_to_max_length: bool = False,
    profile: bool = False,
) -> Dict[str, Any]:
    """CSV 流式评估：逐行读取 CSV -> 统一结构 -> IPC 前缀匹配打标 -> 批量分词 -> 推理 -> 增量写出/累计指标。"""
    if pp_process_csv_file_stream is None or pp_normalize_single_ipc is None:
        raise RuntimeError('预处理流式函数不可用，请确认 preprocess.py 可导入')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(model_dir, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)  # type: ignore
    _ensure_tok_model_compat(tokenizer, model)
    model.eval()

    valid_norm = sorted({pp_normalize_single_ipc(v) for v in (valid_labels_cfg or []) if v}, key=lambda x: (-len(x), x))
    local_opened = False
    writer = shared_writer
    if writer is None and save_predictions:
        writer = open(save_predictions, 'w', encoding='utf-8')
        local_opened = True

    tp=tn=fp=fn=0
    n_total = 0
    ipc_agg: Dict[str, Dict[str, int]] = {}
    idx = 0

    def update_ipc(ipc_list: List[str], labels_b: List[int], preds_b: List[int]):
        if not ipc_list:
            return
        for i, k in enumerate(ipc_list):
            key = k or ''
            a = ipc_agg.get(key)
            if a is None:
                a = {'total':0,'label_pos':0,'pred_pos':0,'correct':0,'tp_pos':0}
                ipc_agg[key] = a
            a['total'] += 1
            lab = labels_b[i]
            pred = preds_b[i]
            if lab == 1:
                a['label_pos'] += 1
            if pred == 1:
                a['pred_pos'] += 1
            if pred == lab:
                a['correct'] += 1
            if pred == 1 and lab == 1:
                a['tp_pos'] += 1

    # 批缓冲
    buf_texts: List[str] = []
    buf_labels: List[int] = []
    buf_ipc: List[str] = []

    try:
        # 逐行流式读取 CSV
        line_index = 0
        import time
        tok_t = fwd_t = io_t = 0.0
        for rec in tqdm(pp_process_csv_file_stream(csv_path, remove_keywords or []), desc='推理(流式CSV)', unit='行'):
            try:
                # 文件内按行分片（如启用）
                if shard_within_file and world_size > 1:
                    if (line_index % world_size) != rank:
                        line_index += 1
                        continue
                    line_index += 1
                text = rec.get('text') or ''
                if not text:
                    continue
                ipc_code = pp_normalize_single_ipc(rec.get('ipc',''))
                lab = 0
                for pref in valid_norm:
                    if ipc_code.startswith(pref):
                        lab = 1
                        break
                buf_texts.append(text)
                buf_labels.append(int(lab))
                buf_ipc.append(ipc_code)
            except Exception:
                continue

            if len(buf_labels) >= batch_size:
                t0 = time.perf_counter()
                enc = tokenizer(
                    buf_texts,
                    truncation=True,
                    max_length=max_length,
                    padding=('max_length' if pad_to_max_length else True),
                    return_tensors='pt'
                )
                tok_t += (time.perf_counter() - t0)
                inputs = {k: v.to(device) for k,v in enc.items()}
                with torch.inference_mode():
                    t1 = time.perf_counter()
                    with _autocast_ctx(device, amp_precision):
                        logits = model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1)
                    fwd_t += (time.perf_counter() - t1)
                prob1_b = probs[:,1].detach().cpu().tolist()
                preds_b = [1 if p > threshold else 0 for p in prob1_b]
                for j, pred in enumerate(preds_b):
                    gt = buf_labels[j]
                    if pred == 1 and gt == 1:
                        tp += 1
                    elif pred == 0 and gt == 0:
                        tn += 1
                    elif pred == 1 and gt == 0:
                        fp += 1
                    else:
                        fn += 1
                n_total += len(preds_b)
                update_ipc(buf_ipc, buf_labels, preds_b)
                if writer is not None:
                    lines = []
                    for j, pred in enumerate(preds_b):
                        rec_out = {
                            'index': index_offset + idx + j,
                            'pred': int(pred),
                            'prob_valid': float(prob1_b[j]),
                            'label': int(buf_labels[j]),
                            'text': buf_texts[j],
                            'ipc': buf_ipc[j],
                        }
                        if source_name:
                            rec_out['source'] = source_name
                        lines.append(json.dumps(rec_out, ensure_ascii=False) + '\n')
                    t2 = time.perf_counter(); writer.write(''.join(lines)); io_t += (time.perf_counter() - t2)
                idx += len(preds_b)
                buf_texts.clear(); buf_labels.clear(); buf_ipc.clear()
                if torch.cuda.is_available() and empty_cache_interval > 0 and (idx // max(1, batch_size)) % empty_cache_interval == 0:
                    torch.cuda.empty_cache()

        # 尾批
        if buf_labels:
            t0 = time.perf_counter()
            enc = tokenizer(
                buf_texts,
                truncation=True,
                max_length=max_length,
                padding=('max_length' if pad_to_max_length else True),
                return_tensors='pt'
            )
            tok_t += (time.perf_counter() - t0)
            inputs = {k: v.to(device) for k,v in enc.items()}
            with torch.inference_mode():
                t1 = time.perf_counter()
                with _autocast_ctx(device, amp_precision):
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                fwd_t += (time.perf_counter() - t1)
            prob1_b = probs[:,1].detach().cpu().tolist()
            preds_b = [1 if p > threshold else 0 for p in prob1_b]
            for j, pred in enumerate(preds_b):
                gt = buf_labels[j]
                if pred == 1 and gt == 1:
                    tp += 1
                elif pred == 0 and gt == 0:
                    tn += 1
                elif pred == 1 and gt == 0:
                    fp += 1
                else:
                    fn += 1
            n_total += len(preds_b)
            update_ipc(buf_ipc, buf_labels, preds_b)
            if writer is not None:
                lines = []
                for j, pred in enumerate(preds_b):
                    rec_out = {
                        'index': index_offset + idx + j,
                        'pred': int(pred),
                        'prob_valid': float(prob1_b[j]),
                        'label': int(buf_labels[j]),
                        'text': buf_texts[j],
                        'ipc': buf_ipc[j],
                    }
                    if source_name:
                        rec_out['source'] = source_name
                    lines.append(json.dumps(rec_out, ensure_ascii=False) + '\n')
                t2 = time.perf_counter(); writer.write(''.join(lines)); io_t += (time.perf_counter() - t2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if local_opened and writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        _cuda_cleanup()

    if local_opened and writer is not None:
        writer.close()

    # 汇总
    accuracy = (tp + tn) / max(1, (tp+tn+fp+fn))
    precision = tp / max(1, (tp+fp))
    recall = tp / max(1, (tp+fn))
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    eval_res = EvalResult(accuracy, precision, recall, f1, tp, tn, fp, fn, n_total)

    ipc_stats: List[Dict[str, Any]] = []
    for k, a in ipc_agg.items():
        total_k = a['total']
        prec_k = a['tp_pos']/a['pred_pos'] if a['pred_pos'] else 0.0
        rec_k = a['tp_pos']/a['label_pos'] if a['label_pos'] else 0.0
        f1_k = (2*prec_k*rec_k/(prec_k+rec_k)) if (prec_k+rec_k)>0 else 0.0
        ipc_stats.append({
            'ipc': k,
            'total': total_k,
            'accuracy': a['correct']/total_k if total_k else 0.0,
            'label_pos': a['label_pos'],
            'pred_pos': a['pred_pos'],
            'tp_pos': a['tp_pos'],
            'precision_pos': prec_k,
            'recall_pos': rec_k,
            'f1_pos': f1_k,
        })
    ipc_stats.sort(key=lambda x: x['total'], reverse=True)

    report_text = (
        f"Streaming CSV report (binary)\n"
        f"samples={n_total} accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}\n"
        f"tn={tn} fp={fp} fn={fn} tp={tp}\n"
    )

    out = {
        'metrics': eval_res.to_dict(),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'classification_report': report_text,
        'samples': int(n_total),
        'model': str(model_dir),
        'input': str(csv_path),
        'already_tokenized': False,
        'max_length': int(max_length),
    'threshold': float(threshold),
        'ipc_stats': ipc_stats,
    'predictions_written': int(n_total),
    }
    if profile:
        total_p = (tok_t + fwd_t + io_t) if ('tok_t' in locals()) else None
        if total_p and total_p > 0:
            print(f"[性能] tokenization {tok_t:.3f}s ({tok_t/total_p:.1%}), forward {fwd_t:.3f}s ({fwd_t/total_p:.1%}), io {io_t:.3f}s ({io_t/total_p:.1%})")
    return out


def evaluate(model_dir: str,
             input: Optional[str],
             batch_size: int,
             max_length: int,
             threshold: float,
             already_tokenized: bool,
             text_key: str,
             label_key: str,
             save_predictions: Optional[str] = None,
             in_memory_texts: Optional[List[str]] = None,
             in_memory_labels: Optional[List[int]] = None,
             ipc_key: str = 'ipc',
             in_memory_ipcs: Optional[List[str]] = None,
             shared_writer: Optional[Any] = None,
             source_name: Optional[str] = None,
             index_offset: int = 0,
             amp_precision: str = 'auto',
             empty_cache_interval: int = 200,
             pad_to_max_length: bool = False,
             profile: bool = False,
             ) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(model_dir, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    _ensure_tok_model_compat(tokenizer, model)
    # 赋值回 model 以兼容部分类型检查器
    model = model.to(device)  # type: ignore
    model.eval()

    input_ids_list: List[List[int]] = []  # 为类型检查预先定义
    attn_list: List[List[int]] = []
    ipc: Optional[List[str]] = None
    ids_list: Optional[List[Any]] = None
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
            ids_list = bundle.get('ids')
            texts = None
        else:
            data = load_text_file(input, text_key=text_key, label_key=label_key, ipc_key=ipc_key)
            texts = data['texts']
            labels = data['labels']
            ipc = data.get('ipc')
            ids_list = data.get('ids')

    n = len(labels)
    print(f'[加载] 样本数: {n}  已分词: {already_tokenized}')

    preds: List[int] = []
    prob_1: List[float] = []
    id2label = model.config.id2label or {0: 'false', 1: 'true'}

    try:
        import time
        tok_t = fwd_t = 0.0
        for start, end in tqdm(list(batch_iter(n, batch_size)), desc='推理'):
            if already_tokenized:
                batch_ids = input_ids_list[start:end]
                batch_attn = attn_list[start:end]
                max_len_batch = max(len(x) for x in batch_ids) if batch_ids else 1
                if max_len_batch > max_length:
                    max_len_batch = max_length
                input_ids_tensor = torch.full((end-start, max_len_batch), tokenizer.pad_token_id or 0, dtype=torch.long)
                attn_tensor = torch.zeros((end-start, max_len_batch), dtype=torch.long)
                for i, (ids, attn) in enumerate(zip(batch_ids, batch_attn)):
                    ids = ids[:max_len_batch]
                    attn = attn[:max_len_batch]
                    input_ids_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attn_tensor[i, :len(attn)] = torch.tensor(attn, dtype=torch.long)
                inputs = {
                    'input_ids': input_ids_tensor.to(device),
                    'attention_mask': attn_tensor.to(device),
                }
            else:
                batch_texts = texts[start:end]  # type: ignore
                t0 = time.perf_counter()
                enc = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=max_length,
                    padding=('max_length' if pad_to_max_length else True),
                    return_tensors='pt'
                )
                tok_t += (time.perf_counter() - t0)
                inputs = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                t1 = time.perf_counter()
                with _autocast_ctx(device, amp_precision):
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                fwd_t += (time.perf_counter() - t1)
            p1 = probs[:, 1].detach().cpu()
            pred_b = (p1 > threshold).to(torch.int64).tolist()
            preds.extend(pred_b)
            prob_1.extend(p1.tolist())
            if torch.cuda.is_available() and empty_cache_interval > 0 and ((end // max(1, batch_size)) % empty_cache_interval == 0):
                torch.cuda.empty_cache()
    finally:
        _cuda_cleanup()

    # Metrics
    import numpy as np
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    accuracy = float((labels_arr == preds_arr).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(labels_arr, preds_arr, average='binary', zero_division=0)
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    eval_res = EvalResult(accuracy, float(precision), float(recall), float(f1), int(tp), int(tn), int(fp), int(fn), int(len(labels)))

    report = classification_report(labels_arr, preds_arr, target_names=[id2label.get(0, '0'), id2label.get(1, '1')], zero_division=0)

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
                a = {'total': 0, 'label_pos': 0, 'pred_pos': 0, 'correct': 0, 'tp_pos': 0}
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
            prec = a['tp_pos'] / a['pred_pos'] if a['pred_pos'] else 0.0
            rec = a['tp_pos'] / a['label_pos'] if a['label_pos'] else 0.0
            f1_k = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            ipc_stats.append({
                'ipc': k,
                'total': total_k,
                'accuracy': a['correct'] / total_k if total_k else 0.0,
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
        'threshold': float(threshold),
        'ipc_stats': ipc_stats,
        'predictions_written': int(n),
    }
    if profile and not already_tokenized:
        total = tok_t + fwd_t
        if total > 0:
            print(f"[性能] tokenization {tok_t:.3f}s ({tok_t/total:.1%}), forward {fwd_t:.3f}s ({fwd_t/total:.1%})")
    # 写预测：共享 writer 优先，否则写入到 save_predictions
    if shared_writer is not None or save_predictions:
        local_opened = False
        f = shared_writer
        if f is None and save_predictions:
            f = open(save_predictions, 'w', encoding='utf-8')
            local_opened = True
        try:
            assert f is not None
            for i, (pred, prob) in enumerate(zip(preds, prob_1)):
                rec = {
                    'index': index_offset + i,
                    'pred': int(pred),
                    'prob_valid': float(prob),
                    'label': int(labels[i])
                }
                if not already_tokenized and texts is not None:
                    rec['text'] = texts[i]
                if ipc is not None and len(ipc) == n:
                    rec['ipc'] = ipc[i]
                if source_name:
                    rec['source'] = source_name
                if ids_list is not None and len(ids_list) == n:
                    rec['id'] = ids_list[i]
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        finally:
            if local_opened and f is not None:
                f.close()
    return out


def _write_ipc_and_metrics(res: Dict[str, Any], ipc_top_k: int, ipc_min_total: int, ipc_summary_text_path: Optional[str], metrics_output_path: str):
    """基于返回结果写出 IPC 文本汇总与 metrics.json。"""
    # 控制台输出总体准确率
    try:
        acc_val = float((res.get('metrics') or {}).get('accuracy', 0.0))
        print(f"[指标] 总体准确率(accuracy): {acc_val:.4f}")
    except Exception:
        pass
    ipc_stats = res.get('ipc_stats') or []
    if ipc_stats and ipc_summary_text_path:
        top_k = max(1, ipc_top_k)
        min_total = max(1, ipc_min_total)
        enriched_pred = []  # (ipc,total,pred_pos,ratio_pred)
        enriched_label = []  # (ipc,total,label_pos,ratio_label)
        for row in ipc_stats:
            total = int(row.get('total',0))
            if total <= 0:
                continue
            pred_pos = int(row.get('pred_pos',0))
            label_pos = int(row.get('label_pos',0))
            enriched_pred.append((row.get('ipc',''), total, pred_pos, pred_pos/total if total else 0.0))
            enriched_label.append((row.get('ipc',''), total, label_pos, label_pos/total if total else 0.0))

        distinct_count = len(enriched_pred)
        total_samples = sum(t for _,t,_,_ in enriched_pred)
        total_positive_pred = sum(p for _,_,p,_ in enriched_pred)
        total_positive_label = sum(p for _,_,p,_ in enriched_label)
        overall_ratio_pred = total_positive_pred/total_samples if total_samples else 0.0
        overall_ratio_label = total_positive_label/total_samples if total_samples else 0.0
        filtered_pred = [e for e in enriched_pred if e[1] >= min_total]
        filtered_label = [e for e in enriched_label if e[1] >= min_total]
        max_ratio_pred = max(filtered_pred, key=lambda x: (x[3], x[1])) if filtered_pred else None
        max_ratio_label = max(filtered_label, key=lambda x: (x[3], x[1])) if filtered_label else None
        top_ratio_pred = sorted(filtered_pred, key=lambda x: (-x[3], -x[1], x[0]))[:top_k]
        top_ratio_label = sorted(filtered_label, key=lambda x: (-x[3], -x[1], x[0]))[:top_k]
        top_count_pred = sorted(enriched_pred, key=lambda x: (-x[1], -x[3], x[0]))[:top_k]

        summary = {
            'distinct_ipc': distinct_count,
            'total_samples': total_samples,
            # 兼容旧字段: 默认使用预测
            'total_positive': total_positive_pred,
            'overall_positive_ratio': overall_ratio_pred,
            'positive_source': 'prediction',
            'prediction_summary': {
                'total_positive': total_positive_pred,
                'overall_positive_ratio': overall_ratio_pred,
                'max_positive_ratio_ipc': {
                    'ipc': max_ratio_pred[0],
                    'total': max_ratio_pred[1],
                    'positive': max_ratio_pred[2],
                    'positive_ratio': max_ratio_pred[3],
                } if max_ratio_pred else None,
                'top_positive_ratio': [
                    {
                        'ipc': ipc,
                        'total': tot,
                        'positive': pos,
                        'positive_ratio': ratio,
                    } for ipc,tot,pos,ratio in top_ratio_pred
                ],
            },
            'label_summary': {
                'total_positive': total_positive_label,
                'overall_positive_ratio': overall_ratio_label,
                'max_positive_ratio_ipc': {
                    'ipc': max_ratio_label[0],
                    'total': max_ratio_label[1],
                    'positive': max_ratio_label[2],
                    'positive_ratio': max_ratio_label[3],
                } if max_ratio_label else None,
                'top_positive_ratio': [
                    {
                        'ipc': ipc,
                        'total': tot,
                        'positive': pos,
                        'positive_ratio': ratio,
                    } for ipc,tot,pos,ratio in top_ratio_label
                ],
            },
            'top_total': [
                {
                    'ipc': ipc,
                    'total': tot,
                    'prediction_positive': pos,
                    'prediction_positive_ratio': ratio,
                } for ipc,tot,pos,ratio in top_count_pred
            ],
            'comparison': {
                'overall_positive_ratio_diff': overall_ratio_pred - overall_ratio_label,
                'overall_positive_ratio_abs_diff': abs(overall_ratio_pred - overall_ratio_label),
                'total_positive_diff': total_positive_pred - total_positive_label,
            },
            'min_total_filter': min_total,
            'top_k': top_k,
            'gt_total_positive': total_positive_label,
            'gt_overall_positive_ratio': overall_ratio_label,
        }
        if top_count_pred:
            summary['max_total_ipc'] = {
                'ipc': top_count_pred[0][0],
                'total': top_count_pred[0][1],
                'prediction_positive': top_count_pred[0][2],
                'prediction_positive_ratio': top_count_pred[0][3],
            }
        res['ipc_summary'] = summary

        try:
            lines = []
            try:
                lines.append(f"总体准确率: {float((res.get('metrics') or {}).get('accuracy', 0.0)):.4f}")
            except Exception:
                pass
            lines.append(f"统计了 {summary['distinct_ipc']} 个不同的 IPC 代码")
            lines.append(f"总样本数: {summary['total_samples']}")
            lines.append(f"[预测] 总正例数: {summary['total_positive']}  占比: {summary['overall_positive_ratio']:.4f}")
            lines.append(f"[真实] 总正例数: {summary['gt_total_positive']}  占比: {summary['gt_overall_positive_ratio']:.4f}")
            comp = summary.get('comparison', {})
            if comp:
                lines.append(f"总体占比差(预测-真实): {comp.get('overall_positive_ratio_diff',0):.4f}")
            pred_max = summary.get('prediction_summary', {}).get('max_positive_ratio_ipc')
            label_max = summary.get('label_summary', {}).get('max_positive_ratio_ipc')
            if pred_max:
                lines.append(f"[预测] 最高占比 IPC: {pred_max['ipc']} ({pred_max['positive_ratio']:.4f}) ({pred_max['positive']}/{pred_max['total']})")
            if label_max:
                lines.append(f"[真实] 最高占比 IPC: {label_max['ipc']} ({label_max['positive_ratio']:.4f}) ({label_max['positive']}/{label_max['total']})")
            if summary.get('max_total_ipc'):
                m2 = summary['max_total_ipc']
                lines.append(f"样本最多的 IPC: {m2['ipc']} ({m2['total']} 条, 预测正例占比 {m2['prediction_positive_ratio']:.4f})")
            lines.append("")
            lines.append(f"[预测] 正例占比前 {summary['top_k']} 名 (min_total={summary['min_total_filter']}):")
            for e in summary['prediction_summary']['top_positive_ratio']:
                lines.append(f"  {e['ipc']}: {e['positive_ratio']:.4f} ({e['positive']}/{e['total']})")
            lines.append("")
            lines.append(f"[真实] 正例占比前 {summary['top_k']} 名 (min_total={summary['min_total_filter']}):")
            for e in summary['label_summary']['top_positive_ratio']:
                lines.append(f"  {e['ipc']}: {e['positive_ratio']:.4f} ({e['positive']}/{e['total']})")
            lines.append("")
            lines.append(f"样本数前 {summary['top_k']} 名:")
            for e in summary['top_total']:
                lines.append(f"  {e['ipc']}: {e['total']} 条 (预测正例占比 {e['prediction_positive_ratio']:.4f})")
            os.makedirs(os.path.dirname(ipc_summary_text_path), exist_ok=True)
            with open(ipc_summary_text_path, 'w', encoding='utf-8') as ftxt:
                ftxt.write('\n'.join(lines))
            print(f"[输出] IPC 文本汇总写入 {ipc_summary_text_path}")
        except Exception as e:
            print(f"[警告] 写入 IPC 文本汇总失败: {e}")

    # 写 metrics.json
    metrics_json = json.dumps(res, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w', encoding='utf-8') as f:
        f.write(metrics_json)
    print(f"[输出] 指标写入 {metrics_output_path}")


def _with_stem_suffix(path: str, stem: str) -> str:
    """在文件名上加入 __{stem} 后缀 (扩展名前)。"""
    d = os.path.dirname(path)
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    if not ext:
        # 若无扩展名，默认作为目录/文件前缀处理
        return os.path.join(path, f"{stem}")
    return os.path.join(d, f"{name}__{stem}{ext}")


def _expand_inputs(pattern: str) -> List[str]:
    wc = set('*?[]')
    if any(c in pattern for c in wc):
        matches = glob.glob(pattern, recursive=True)
        # 仅保留文件
        return sorted([os.path.abspath(m) for m in matches if os.path.isfile(m)])
    # 非通配符，直接返回存在的路径
    if os.path.isfile(pattern):
        return [os.path.abspath(pattern)]
    return []


def _merge_prediction_files(pred_paths: List[Tuple[str, str]], out_path: str, delete_sources: bool = False):
    """将多个预测 jsonl 合并为一个，添加 source 字段标识来源文件。
    pred_paths: 列表 (source_name, file_path)
    """
    if not pred_paths:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fout:
        for source_name, p in pred_paths:
            if not p or not os.path.exists(p):
                continue
            with open(p, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    obj['source'] = source_name
                    fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
    if delete_sources:
        for _, p in pred_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def _ipc_stats_from_predictions(pred_path: str) -> List[Dict[str, Any]]:
    """从合并后的 predictions.jsonl 计算 IPC 统计。
    返回列表元素结构与 evaluate/evaluate_stream 的 ipc_stats 一致。
    """
    agg: Dict[str, Dict[str, int]] = {}
    if not pred_path or not os.path.exists(pred_path):
        return []
    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                try:
                    lab = int(obj.get('label', 0))
                except Exception:
                    continue
                try:
                    pred = int(obj.get('pred', 0))
                except Exception:
                    pred = 0
                ipc = obj.get('ipc') or ''
                a = agg.get(ipc)
                if a is None:
                    a = {'total':0,'label_pos':0,'pred_pos':0,'correct':0,'tp_pos':0}
                    agg[ipc] = a
                a['total'] += 1
                if lab == 1:
                    a['label_pos'] += 1
                if pred == 1:
                    a['pred_pos'] += 1
                if pred == lab:
                    a['correct'] += 1
                if pred == 1 and lab == 1:
                    a['tp_pos'] += 1
    except Exception:
        return []

    ipc_stats: List[Dict[str, Any]] = []
    for k, a in agg.items():
        total_k = a['total']
        prec_k = a['tp_pos']/a['pred_pos'] if a['pred_pos'] else 0.0
        rec_k = a['tp_pos']/a['label_pos'] if a['label_pos'] else 0.0
        f1_k = (2*prec_k*rec_k/(prec_k+rec_k)) if (prec_k+rec_k)>0 else 0.0
        ipc_stats.append({
            'ipc': k,
            'total': total_k,
            'accuracy': a['correct']/total_k if total_k else 0.0,
            'label_pos': a['label_pos'],
            'pred_pos': a['pred_pos'],
            'tp_pos': a['tp_pos'],
            'precision_pos': prec_k,
            'recall_pos': rec_k,
            'f1_pos': f1_k,
        })
    ipc_stats.sort(key=lambda x: x['total'], reverse=True)
    return ipc_stats


def main():
    parser = argparse.ArgumentParser(description='评估脚本: 支持 jsonl/json 以及 csv (直接预处理内存推理)')
    parser.add_argument('--model', required=True, help='模型目录 (含 config.json, tokenizer)')
    parser.add_argument('--input', required=True, help='标注数据 (.jsonl/.json 或 .csv)，支持通配符 (如 data/*.jsonl)')
    parser.add_argument('--already-tokenized', action='store_true', help='输入为已分词 (含 input_ids/attention_mask) 的 jsonl/json')
    parser.add_argument('--stream', action='store_true', help='启用流式：逐行读取、批量推理、增量写出，降低内存占用 (仅支持 JSONL)')
    parser.add_argument('--text-key', default='text')
    parser.add_argument('--label-key', default='label')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.5, help='正类判定阈值，prob(1) > threshold 判为 1')
    parser.add_argument('--gpus', default=None, help='指定可见 GPU (如 "0" 或 "0,1")')
    parser.add_argument('--config', default='config/config.json', help='当输入为 CSV 时加载的配置文件 (含 preprocess_config.valid_labels/remove_keywords)')
    parser.add_argument('--ipc-key', default='ipc', help='IPC 字段名称 (用于统计)')
    parser.add_argument('--ipc-top-k', type=int, default=10, help='IPC 汇总中前K名 (比例/数量)')
    parser.add_argument('--ipc-min-total', type=int, default=1, help='计算最高占比/前K占比列表时的最小样本数过滤 (默认不过滤)')
    parser.add_argument('--output-dir', default='outputs/metrics', help='将结果统一输出到该目录 (会生成 predictions.jsonl / metrics.json / ipc_summary.txt)')
    parser.add_argument('--precision', default='auto', choices=['auto','fp32','fp16','bf16'], help='推理精度：auto 会在 CUDA 上使用混合精度')
    parser.add_argument('--empty-cache-interval', type=int, default=200, help='每多少个批次/步清一次 CUDA 缓存；<=0 表示不清理')
    parser.add_argument('--pad-to-max-length', action='store_true', help='分词时按固定 max_length 对齐填充，提升批内对齐效率')
    parser.add_argument('--profile', action='store_true', help='打印简单的 tokenization/forward/io 时间占比')
    args = parser.parse_args()

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f"设置 CUDA_VISIBLE_DEVICES={args.gpus}")
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    gpu_env_diag()
    is_dist, rank, world_size, local_rank = _maybe_init_dist()
    if is_dist:
        print(f"[分布式] world_size={world_size} rank={rank} local_rank={local_rank}")

    # 统一输出目录（必填）
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    # 为复用后续逻辑，直接在 args 上挂载统一的输出路径
    args.save_predictions = os.path.join(out_dir, 'predictions.jsonl')
    args.metrics_output = os.path.join(out_dir, 'metrics.json')
    args.ipc_summary_text = os.path.join(out_dir, 'ipc_summary.txt')
    print(f"[输出] 统一输出目录: {out_dir}")

    # 扩展输入 (通配符)
    inputs = _expand_inputs(args.input)
    if not inputs:
        # 若未匹配到，用原样路径尝试 (兼容非文件，如管道，尽管这里主要针对文件)
        inputs = [args.input]
    print(f"[输入] 匹配到 {len(inputs)} 个文件")

    multi = len(inputs) > 1
    # 基于 rank 的文件级切分：若文件数量 >= world_size，则每个 rank 处理一个子集
    if world_size > 1 and len(inputs) >= world_size:
        inputs_rank = [p for i,p in enumerate(inputs) if (i % world_size) == rank]
    else:
        inputs_rank = inputs

    file_results: List[Dict[str, Any]] = []
    # 聚合器
    agg_tp=agg_tn=agg_fp=agg_fn=0
    agg_samples = 0
    agg_ipc: Dict[str, Dict[str, int]] = {}
    pred_parts: List[Tuple[str, str]] = []  # (source_name, pred_path)
    # 统一预测输出
    shared_pred_writer = None
    global_index = 0
    final_pred_path = args.save_predictions if args.save_predictions else None
    if final_pred_path:
        if world_size > 1:
            # 分布式：每 rank 写临时文件，稍后 rank0 合并
            d = os.path.dirname(final_pred_path)
            n, ext = os.path.splitext(os.path.basename(final_pred_path))
            pred_out_path_for_rank = os.path.join(d, f"{n}__rank{rank}{ext}")
        else:
            # 单进程：开启一个共享 writer
            os.makedirs(os.path.dirname(final_pred_path) or '.', exist_ok=True)
            shared_pred_writer = open(final_pred_path, 'w', encoding='utf-8')
            pred_out_path_for_rank = final_pred_path
    else:
        pred_out_path_for_rank = None

    for i_path in inputs_rank:
        print(f"\n=== 处理输入: {i_path} ===")
        stem = os.path.splitext(os.path.basename(i_path))[0]
        # 仅统一输出：不再生成每文件的 metrics/ipc 文本
        metrics_path = None
        pred_path = pred_out_path_for_rank
        ipc_text_path = None

        # 选择评估路径
        if i_path.lower().endswith('.csv'):
            if args.stream:
                if pp_load_config is None or pp_process_csv_file_stream is None or pp_normalize_single_ipc is None:
                    raise RuntimeError('无法导入 preprocess 中的流式函数，请确保 preprocess.py 可用')
                if not os.path.exists(args.config):
                    raise FileNotFoundError(f'配置文件不存在: {args.config}')
                cfg_all = pp_load_config(args.config)
                if 'preprocess_config' not in cfg_all:
                    raise ValueError('配置文件缺少 preprocess_config 节点')
                pp_cfg = cfg_all['preprocess_config']
                res = evaluate_stream_csv(
                    model_dir=args.model,
                    csv_path=i_path,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    threshold=args.threshold,
                    text_key=args.text_key,
                    label_key=args.label_key,
                    save_predictions=pred_path,
                    ipc_key=args.ipc_key,
                    remove_keywords=pp_cfg.get('remove_keywords') or [],
                    valid_labels_cfg=pp_cfg.get('valid_labels') or [],
                    shard_within_file=(world_size > 1 and len(inputs) < world_size),
                    rank=rank,
                    world_size=world_size,
                    shared_writer=shared_pred_writer,
                    source_name=stem,
                    index_offset=global_index,
                    amp_precision=args.precision,
                    empty_cache_interval=args.empty_cache_interval,
                    pad_to_max_length=args.pad_to_max_length,
                    profile=args.profile,
                )
            else:
                if pp_load_config is None or pp_process_csv_file is None or pp_normalize_single_ipc is None:
                    raise RuntimeError('无法导入 preprocess 中的函数，请确保 preprocess.py 存在且可导入')
                if not os.path.exists(args.config):
                    raise FileNotFoundError(f'配置文件不存在: {args.config}')
                cfg_all = pp_load_config(args.config)
                if 'preprocess_config' not in cfg_all:
                    raise ValueError('配置文件缺少 preprocess_config 节点')
                pp_cfg = cfg_all['preprocess_config']
                remove_keywords = pp_cfg.get('remove_keywords') or []
                valid_labels_cfg = pp_cfg.get('valid_labels') or []
                print(f'[CSV] 读取并预处理: {i_path}')
                records = pp_process_csv_file(i_path, remove_keywords)
                print(f'[CSV] 原始记录数: {len(records)}  有效前缀候选: {len(valid_labels_cfg)}')
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
                    threshold=args.threshold,
                    already_tokenized=False,
                    text_key=args.text_key,
                    label_key=args.label_key,
                    save_predictions=pred_path,
                    in_memory_texts=texts_mem,
                    in_memory_labels=labels_mem,
                    ipc_key=args.ipc_key,
                    in_memory_ipcs=ipcs_mem,
                    shared_writer=shared_pred_writer,
                    source_name=stem,
                    index_offset=global_index,
                    amp_precision=args.precision,
                    empty_cache_interval=args.empty_cache_interval,
                    pad_to_max_length=args.pad_to_max_length,
                    profile=args.profile,
                )
        else:
            # JSON/JSONL
            if args.stream:
                res = evaluate_stream(
                    model_dir=args.model,
                    input=i_path,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    threshold=args.threshold,
                    already_tokenized=args.already_tokenized,
                    text_key=args.text_key,
                    label_key=args.label_key,
                    save_predictions=pred_path,
                    ipc_key=args.ipc_key,
                    shard_within_file=(world_size > 1 and len(inputs) < world_size),
                    rank=rank,
                    world_size=world_size,
                    shared_writer=shared_pred_writer,
                    source_name=stem,
                    index_offset=global_index,
                    amp_precision=args.precision,
                    empty_cache_interval=args.empty_cache_interval,
                    pad_to_max_length=args.pad_to_max_length,
                    profile=args.profile,
                )
            else:
                res = evaluate(
                    model_dir=args.model,
                    input=i_path,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    threshold=args.threshold,
                    already_tokenized=args.already_tokenized,
                    text_key=args.text_key,
                    label_key=args.label_key,
                    save_predictions=pred_path,
                    ipc_key=args.ipc_key,
                    shared_writer=shared_pred_writer,
                    source_name=stem,
                    index_offset=global_index,
                    amp_precision=args.precision,
                    empty_cache_interval=args.empty_cache_interval,
                    pad_to_max_length=args.pad_to_max_length,
                    profile=args.profile,
                )

        # 不进行逐文件指标写出

        # 累积到聚合
        cm = res.get('confusion_matrix') or {}
        agg_tn += int(cm.get('tn', 0))
        agg_fp += int(cm.get('fp', 0))
        agg_fn += int(cm.get('fn', 0))
        agg_tp += int(cm.get('tp', 0))
        agg_samples += int(res.get('samples', 0))
        # 合并 IPC 统计
        for row in (res.get('ipc_stats') or []):
            k = row.get('ipc', '')
            if not k and k != '':
                continue
            a = agg_ipc.get(k)
            if a is None:
                a = {'total':0,'label_pos':0,'pred_pos':0,'correct':0,'tp_pos':0}
                agg_ipc[k] = a
            a['total'] += int(row.get('total',0))
            a['label_pos'] += int(row.get('label_pos',0))
            a['pred_pos'] += int(row.get('pred_pos',0))
            a['correct'] += int(row.get('correct',0))
            a['tp_pos'] += int(row.get('tp_pos',0))

        file_results.append({
            'input': i_path,
            'samples': int(res.get('samples', 0)),
            'metrics': res.get('metrics'),
        })
        try:
            global_index += int(res.get('predictions_written', res.get('samples', 0)))
        except Exception:
            global_index += int(res.get('samples', 0))
        if world_size > 1 and pred_out_path_for_rank:
            pred_parts.append((stem, pred_out_path_for_rank))

    # 仅输出一次聚合指标
    if True:
        acc = (agg_tp + agg_tn) / max(1, (agg_tp+agg_tn+agg_fp+agg_fn))
        prec = agg_tp / max(1, (agg_tp+agg_fp))
        rec = agg_tp / max(1, (agg_tp+agg_fn))
        f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        res_all = {
            'metrics': EvalResult(acc, prec, rec, f1, agg_tp, agg_tn, agg_fp, agg_fn, agg_samples).to_dict(),
            'confusion_matrix': {'tn': int(agg_tn), 'fp': int(agg_fp), 'fn': int(agg_fn), 'tp': int(agg_tp)},
            'classification_report': f'Aggregated over {len(inputs)} files',
            'samples': int(agg_samples),
            'model': str(args.model),
            'input': inputs,
            'already_tokenized': bool(args.already_tokenized),
            'max_length': int(args.max_length),
            'threshold': float(args.threshold),
        }
        # 聚合 IPC
        ipc_stats_all: List[Dict[str, Any]] = []
        for k, a in agg_ipc.items():
            total_k = a['total']
            prec_k = a['tp_pos']/a['pred_pos'] if a['pred_pos'] else 0.0
            rec_k = a['tp_pos']/a['label_pos'] if a['label_pos'] else 0.0
            f1_k = (2*prec_k*rec_k/(prec_k+rec_k)) if (prec_k+rec_k)>0 else 0.0
            ipc_stats_all.append({
                'ipc': k,
                'total': total_k,
                'accuracy': a['correct']/total_k if total_k else 0.0,
                'label_pos': a['label_pos'],
                'pred_pos': a['pred_pos'],
                'tp_pos': a['tp_pos'],
                'precision_pos': prec_k,
                'recall_pos': rec_k,
                'f1_pos': f1_k,
            })
        ipc_stats_all.sort(key=lambda x: x['total'], reverse=True)
        res_all['ipc_stats'] = ipc_stats_all
        res_all['files'] = file_results

        if world_size == 1:
            _write_ipc_and_metrics(
                res_all,
                ipc_top_k=args.ipc_top_k,
                ipc_min_total=args.ipc_min_total,
                ipc_summary_text_path=args.ipc_summary_text,
                metrics_output_path=args.metrics_output,
            )

    # 分布式最终合并：仅 rank0 写最终指标，并合并各 rank 的预测到一个文件
    if world_size > 1:
        try:
            dist.barrier()
        except Exception:
            pass
        # 先进行全局指标求和
        try:
            dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            t = torch.tensor([agg_tn, agg_fp, agg_fn, agg_tp, agg_samples], dtype=torch.long, device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            g_tn, g_fp, g_fn, g_tp, g_samples = [int(x) for x in t.tolist()]
        except Exception:
            # 回退：使用本地（不准确）
            g_tn, g_fp, g_fn, g_tp, g_samples = agg_tn, agg_fp, agg_fn, agg_tp, agg_samples
        if rank == 0:
            # 合并各 rank 的预测 -> 生成全量 predictions.jsonl
            merged_pred_path = args.save_predictions if args.save_predictions else None
            if merged_pred_path:
                base = merged_pred_path
                d = os.path.dirname(base)
                n, ext = os.path.splitext(os.path.basename(base))
                parts = []
                for r in range(world_size):
                    p = os.path.join(d, f"{n}__rank{r}{ext}")
                    if os.path.exists(p):
                        parts.append((f"rank{r}", p))
                if parts:
                    _merge_prediction_files(parts, base, delete_sources=True)

            # 基于合并后的预测重建 IPC 统计
            ipc_stats_final: List[Dict[str, Any]] = []
            if merged_pred_path and os.path.exists(merged_pred_path):
                ipc_stats_final = _ipc_stats_from_predictions(merged_pred_path)

            # 计算全局指标并写出（包含 IPC 统计）
            accG = (g_tp + g_tn) / max(1, (g_tp+g_tn+g_fp+g_fn))
            precG = g_tp / max(1, (g_tp+g_fp))
            recG = g_tp / max(1, (g_tp+g_fn))
            f1G = (2*precG*recG)/(precG+recG) if (precG+recG)>0 else 0.0
            _write_ipc_and_metrics(
                {
                    'metrics': EvalResult(accG, precG, recG, f1G, g_tp, g_tn, g_fp, g_fn, g_samples).to_dict(),
                    'confusion_matrix': {'tn': int(g_tn), 'fp': int(g_fp), 'fn': int(g_fn), 'tp': int(g_tp)},
                    'classification_report': f'Aggregated over {world_size} ranks',
                    'samples': int(g_samples),
                    'model': str(args.model),
                    'input': inputs,
                    'already_tokenized': bool(args.already_tokenized),
                    'max_length': int(args.max_length),
                    'threshold': float(args.threshold),
                    'ipc_stats': ipc_stats_final,
                },
                ipc_top_k=args.ipc_top_k,
                ipc_min_total=args.ipc_min_total,
                ipc_summary_text_path=args.ipc_summary_text,
                metrics_output_path=args.metrics_output,
            )
        # 关闭共享 writer（分布式不使用，共享则在非分布式下）
        if shared_pred_writer is not None:
            try:
                shared_pred_writer.close()
            except Exception:
                pass
    else:
        # 非分布式：关闭共享 writer
        if shared_pred_writer is not None:
            try:
                shared_pred_writer.close()
            except Exception:
                pass


if __name__ == '__main__':
    try:
        main()
    finally:
        # 进程退出前做一次全局清理；分布式则销毁进程组
        _cuda_cleanup()
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
