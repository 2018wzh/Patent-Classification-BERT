#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按 IPC 统计总数量、有效数量(label==1)、有效占比，并导出 CSV。
默认输入: dataset/data_origin.jsonl
默认输出: outputs/analysis/ipc_stats.csv

新增: --stream 流式模式 (仅支持 JSONL)，逐行统计以降低内存占用。
"""
import os
import sys
import json
import csv
import argparse
from typing import Dict, Any, Optional

# 尝试使用预处理中的 IPC 规范化函数(可选)
try:
    from preprocess import normalize_single_ipc as pp_normalize_single_ipc  # type: ignore
except Exception:
    pp_normalize_single_ipc = None  # type: ignore

# 尝试引入 tqdm 进度条(可选)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


def normalize_ipc(ipc: Optional[str]) -> str:
    if not ipc:
        return ''
    if pp_normalize_single_ipc:
        try:
            return pp_normalize_single_ipc(ipc) or ''
        except Exception:
            pass
    return str(ipc).strip()


def iter_records(path: str, stream: bool = False):
    """迭代输入记录。
    - JSONL: 始终逐行读取；
    - JSON(list): 在 stream=True 时不支持（需一次性载入），stream=False 时一次性载入列表。
    """
    lower = path.lower()
    if lower.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    elif lower.endswith('.json'):
        if stream:
            raise ValueError('流式模式(--stream)仅支持 JSONL 输入；JSON 数组需关闭 --stream')
        with open(path, 'r', encoding='utf-8') as f:
            try:
                root = json.load(f)
            except Exception:
                root = []
        if isinstance(root, list):
            for obj in root:
                if isinstance(obj, dict):
                    yield obj
    else:
        raise ValueError('仅支持 .jsonl 或 .json 输入')


def read_metadata_total(input_path: str) -> Optional[int]:
    """尝试从与输入同目录的 metadata.json 中读取数据集总数。
    优先读取 counts.total_records；若不存在，回退到 tokenize.observed_count（若存在）。
    """
    try:
        meta_path = os.path.join(os.path.dirname(os.path.abspath(input_path)), 'metadata.json')
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        counts = meta.get('counts') or {}
        if isinstance(counts, dict) and 'total_records' in counts:
            v = counts.get('total_records')
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    pass
        tok = meta.get('tokenize') or {}
        if isinstance(tok, dict) and 'observed_count' in tok:
            v = tok.get('observed_count')
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    pass
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description='IPC 统计 (总数量/有效数量/有效占比) 并导出 CSV')
    parser.add_argument('--input', '-i', default=os.path.join('dataset', 'data_origin.jsonl'), help='输入文件 (.jsonl/.json)')
    parser.add_argument('--output', '-o', default=os.path.join('outputs', 'analysis', 'ipc_stats.csv'), help='输出 CSV 文件路径')
    parser.add_argument('--ipc-key', default='ipc', help='IPC 字段名，默认 ipc')
    parser.add_argument('--label-key', default='label', help='标签字段名，默认 label (1 表示有效)')
    parser.add_argument('--stream', action='store_true', help='流式模式：逐行统计(仅支持 JSONL)')
    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output)

    if not os.path.exists(in_path):
        print(f'[错误] 输入文件不存在: {in_path}', file=sys.stderr)
        sys.exit(2)

    # metadata 预期总数(仅在流式模式下读取)
    meta_total: Optional[int] = None
    if args.stream:
        meta_total = read_metadata_total(in_path)
        if meta_total is not None:
            print(f"[metadata] 预期总数: {meta_total} (来源: {os.path.join(os.path.dirname(in_path), 'metadata.json')})")

    # 统计
    stats: Dict[str, Dict[str, int]] = {}
    total_rows = 0
    with_label = 0
    # 进度条
    pbar = None
    expected_total: Optional[int] = meta_total if args.stream else None
    if tqdm is not None:
        pbar = tqdm(total=expected_total, desc='统计中', unit='条')

    for obj in iter_records(in_path, stream=args.stream):
        total_rows += 1
        if pbar is not None:
            pbar.update(1)
        ipc_raw = obj.get(args.ipc_key)
        ipc = normalize_ipc(ipc_raw)
        if ipc not in stats:
            stats[ipc] = {'total': 0, 'valid': 0}
        stats[ipc]['total'] += 1
        if args.label_key in obj:
            with_label += 1
            lab = obj.get(args.label_key)
            lab_i = 0
            if isinstance(lab, bool):
                lab_i = 1 if lab else 0
            elif lab is None:
                lab_i = 0
            elif isinstance(lab, (int,)):
                lab_i = int(lab)
            elif isinstance(lab, float):
                lab_i = 1 if lab >= 0.5 else 0
            elif isinstance(lab, str):
                try:
                    lab_i = int(lab)
                except Exception:
                    lab_i = 1 if lab.strip().lower() in {'1','true','t','y','yes'} else 0
            else:
                try:
                    lab_i = int(lab)  # 最后尝试
                except Exception:
                    lab_i = 0
            if lab_i == 1:
                stats[ipc]['valid'] += 1

    # 关闭进度条
    if pbar is not None:
        pbar.close()

    # 输出 CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # utf-8-sig 更友好地被 Excel 识别
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ipc', '总数量', '有效数量', '有效占比'])
        # 排序：总数量降序，其次 ipc 升序
        items = sorted(stats.items(), key=lambda kv: (-kv[1]['total'], kv[0]))
        for ipc, d in items:
            total = d['total']
            valid = d['valid']
            ratio = (valid / total) if total else 0.0
            writer.writerow([ipc, total, valid, f'{ratio:.6f}'])

    print(f'[完成] 共读取 {total_rows} 条记录，其中含 label 的记录 {with_label} 条')
    if args.stream and meta_total is not None and meta_total != total_rows:
        print(f"[提示] 实际读取数({total_rows}) 与 metadata 预期总数({meta_total}) 不一致", file=sys.stderr)
    print(f'[输出] 已写入 CSV: {out_path}')


if __name__ == '__main__':
    main()
