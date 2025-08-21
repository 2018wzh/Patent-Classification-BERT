"""读取 config/config.json 中的 pack_config 与 train_config, 自动将指定 jsonl (train/val/test) 打包为 .pt.

不再支持命令行参数。执行:  python pack_dataset.py

config.json 需包含:
    train_config.train_file
    train_config.validation_file (可选)
    以及可选 test 文件 (train_config.test_file 或 dataset/test.jsonl 默认推断)
    pack_config: {
            enable: true,
            max_seq_length: 512 | null,
            pad_token_id: 0,
            suffix: "_packed.pt",
            overwrite: false
    }

输出: 与原 jsonl 同目录, 生成 <basename><suffix>
"""
import os, json
from typing import List, Tuple
import torch
from tqdm import tqdm
import numpy as np


def read_jsonl(path: str):
    data = []
    size = os.path.getsize(path)
    print(f"[read] {path} ({size/1024/1024:.2f} MB)")
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"读取 {os.path.basename(path)}", unit='行'):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data

def pack(samples, max_seq_length: int | None, pad_token_id: int):
    if not samples:
        return {
            'input_ids': torch.empty(0,0,dtype=torch.long),
            'attention_mask': torch.empty(0,0,dtype=torch.long),
            'labels': torch.empty(0,dtype=torch.long),
            'meta': {'num_samples':0,'max_seq_length':0,'avg_seq_length':0,'pad_token_id':pad_token_id}
        }
    if max_seq_length is None:
        max_seq_length = max(len(s['input_ids']) for s in samples)
    N = len(samples)
    input_ids = torch.full((N, max_seq_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((N, max_seq_length), dtype=torch.long)
    labels = torch.zeros((N,), dtype=torch.long)
    lengths = []
    for i, s in enumerate(tqdm(samples, desc='填充打包', unit='样本')):
        ids = s['input_ids'][:max_seq_length]
        attn = s['attention_mask'][:max_seq_length]
        l = len(ids)
        lengths.append(l)
        input_ids[i, :l] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :l] = torch.tensor(attn, dtype=torch.long)
        labels[i] = int(s.get('label') if 'label' in s else s.get('labels', 0))
    meta = {
        'num_samples': N,
        'max_seq_length': max_seq_length,
        'avg_seq_length': sum(lengths)/len(lengths),
        'pad_token_id': pad_token_id,
    }
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'meta': meta}

def process(input_path: str, output_path: str, max_seq_length: int | None, pad_token_id: int, overwrite: bool, enable_memmap: bool):
    if (not overwrite) and os.path.exists(output_path):
        print(f"跳过已存在: {output_path}")
        return
    print(f"读取 {input_path}")
    samples = read_jsonl(input_path)
    print(f"样本数 {len(samples)}  max_seq_length={max_seq_length or 'auto'}")
    bundle = pack(samples, max_seq_length, pad_token_id)
    torch.save(bundle, output_path)
    print(f"保存 {output_path}  meta={bundle['meta']}")
    if enable_memmap:
        base = output_path[:-3]  # 去掉 .pt
        meta = bundle['meta']
        N = meta['num_samples']
        L = meta['max_seq_length']
        # 以只读共享方式创建 .npy (一次写入) -> 训练中用 memmap 读取
        ids_path = base + '.input_ids.npy'
        attn_path = base + '.attention_mask.npy'
        lab_path = base + '.labels.npy'
        meta_path = base + '.memmap_meta.json'
        if overwrite or (not os.path.exists(ids_path)):
            np.save(ids_path, bundle['input_ids'].cpu().numpy())
            np.save(attn_path, bundle['attention_mask'].cpu().numpy())
            np.save(lab_path, bundle['labels'].cpu().numpy())
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'num_samples': int(N),
                    'max_seq_length': int(L),
                    'pad_token_id': int(pad_token_id),
                    'dtype': 'int64'
                }, f, ensure_ascii=False, indent=2)
            print(f"[memmap] 生成: {ids_path}, {attn_path}, {lab_path}")
        else:
            print(f"[memmap] 跳过已存在 memmap 文件 (使用 --overwrite=true 覆盖)")

def load_config(path: str = 'config/config.json'):
    if not os.path.exists(path):
        raise FileNotFoundError(f'配置文件不存在: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    cfg = load_config()
    train_cfg = cfg.get('train_config') or {}
    pack_cfg = cfg.get('pack_config') or {}
    if not pack_cfg.get('enable', True):
        print('pack_config.enable=false, 不执行打包。')
        return
    suffix = pack_cfg.get('suffix', '_packed.pt')
    pad_id = pack_cfg.get('pad_token_id', 0)
    max_seq_len = pack_cfg.get('max_seq_length')  # 可能为 None
    overwrite = bool(pack_cfg.get('overwrite', False))
    enable_memmap = bool(pack_cfg.get('memmap', False))

    candidates: List[Tuple[str, str]] = []
    def add_path(p: str | None):
        if not p:
            return
        if not os.path.exists(p):
            return
        if not p.endswith('.jsonl'):
            return
        outp = p[:-6] + suffix
        candidates.append((p, outp))

    add_path(train_cfg.get('train_file'))
    add_path(train_cfg.get('validation_file'))
    # 试探 test
    if 'test_file' in train_cfg:
        add_path(train_cfg.get('test_file'))
    else:
        guess_test = os.path.join(os.path.dirname(train_cfg.get('train_file','dataset/train.jsonl')), 'test.jsonl')
        if os.path.exists(guess_test):
            add_path(guess_test)

    if not candidates:
        print('未找到可打包的 jsonl 文件。')
        return

    print('=== 开始打包 ===')
    print(f'suffix={suffix} pad_token_id={pad_id} max_seq_length={max_seq_len or "auto"} overwrite={overwrite} memmap={enable_memmap}')
    for inp, outp in tqdm(candidates, desc='文件打包', unit='文件'):
        process(inp, outp, max_seq_len, pad_id, overwrite, enable_memmap)
    print('=== 打包完成 ===')

if __name__ == '__main__':
    main()
