"""读取 config/config.json 中的 pack_config 与 train_config, 自动将指定 jsonl (train/val/test) 打包为 .pt.

可选命令行覆盖(非必须):
    --stream        强制启用流式打包（优先使用 metadata.json 跳过扫描，实现边读边写）
    --memmap        启用 memmap 输出（流式模式下会自动开启）
    --no-pt         仅生成 memmap，不生成 .pt
    --overwrite     覆盖已存在的输出文件

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
import os, json, sys
from typing import List, Tuple
import torch
from tqdm import tqdm
import numpy as np
from numpy.lib.format import open_memmap


def read_jsonl(path: str):
    data = []
    size = os.path.getsize(path)
    print(f"[read] {path} ({size/1024/1024:.2f} MB)")
    # 从 metadata.json 推断总行数，用于进度条
    total = None
    try:
        meta_path = os.path.join(os.path.dirname(path), 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as fmeta:
                meta_json = json.load(fmeta)
            # 若匹配到 split 路径，则使用对应 split 的 N
            split_paths = (meta_json.get('split_paths') or {})
            splits_info = (meta_json.get('splits') or {})
            which = None
            for name, p in split_paths.items():
                try:
                    if os.path.abspath(p) == os.path.abspath(path):
                        which = name
                        break
                except Exception:
                    continue
            if which and isinstance(splits_info.get(which), (int, float)):
                total = int(splits_info[which])
            else:
                # 否则若该文件是 tokenized_jsonl，则用 counts.total_records
                tok_path = ((meta_json.get('paths') or {}).get('tokenized_jsonl'))
                cnts = meta_json.get('counts') or {}
                if tok_path and os.path.abspath(tok_path) == os.path.abspath(path):
                    tr = cnts.get('total_records')
                    if isinstance(tr, (int, float)):
                        total = int(tr)
    except Exception:
        total = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"读取 {os.path.basename(path)}", unit='行', total=total):
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

def _scan_jsonl(input_path: str, max_seq_length: int | None) -> Tuple[int, int]:
    """第一遍扫描: 返回 (样本数N, 序列长度L)。若 max_seq_length 为 None 则以数据最大长度为 L。"""
    N = 0
    maxL = 0
    size_mb = os.path.getsize(input_path) / 1024 / 1024
    print(f"[scan] {input_path} ({size_mb:.2f} MB)")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"扫描 {os.path.basename(input_path)}", unit='行'):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ids = rec.get('input_ids') or []
            N += 1
            if max_seq_length is None:
                if isinstance(ids, list):
                    l = len(ids)
                    if l > maxL:
                        maxL = l
    L = max_seq_length if max_seq_length is not None else maxL
    print(f"[scan] N={N} L={L}")
    return N, L


def _stream_pack_to_memmap(input_path: str, base_out: str, N: int, L: int, pad_token_id: int):
    """第二遍流式写入 memmap .npy: base_out + .input_ids.npy/.attention_mask.npy/.labels.npy 以及 meta json。"""
    ids_path = base_out + '.input_ids.npy'
    attn_path = base_out + '.attention_mask.npy'
    lab_path = base_out + '.labels.npy'
    meta_path = base_out + '.memmap_meta.json'

    print(f"[memmap] 创建 {ids_path}, {attn_path}, {lab_path}")
    ids_mm = open_memmap(ids_path, mode='w+', dtype='int64', shape=(N, L))
    attn_mm = open_memmap(attn_path, mode='w+', dtype='int64', shape=(N, L))
    lab_mm = open_memmap(lab_path, mode='w+', dtype='int64', shape=(N,))

    i = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"打包 {os.path.basename(input_path)}", unit='行', total=N if isinstance(N, int) and N>0 else None):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ids = rec.get('input_ids') or []
            attn = rec.get('attention_mask') or []
            lbl = rec.get('label', rec.get('labels', 0))
            try:
                lbl = int(lbl)
            except Exception:
                lbl = 0
            # 初始化行
            ids_mm[i, :] = pad_token_id
            attn_mm[i, :] = 0
            l = min(len(ids), L)
            if l > 0:
                ids_mm[i, :l] = np.asarray(ids[:l], dtype=np.int64)
                if attn and len(attn) >= l:
                    attn_mm[i, :l] = np.asarray(attn[:l], dtype=np.int64)
                else:
                    attn_mm[i, :l] = 1  # 若未提供 mask, 默认前 l 位为 1
            lab_mm[i] = int(lbl)
            i += 1

    meta = {
        'num_samples': int(N),
        'max_seq_length': int(L),
        'pad_token_id': int(pad_token_id),
        'dtype': 'int64'
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[memmap] 完成写入: {ids_path}, {attn_path}, {lab_path}")
    return ids_path, attn_path, lab_path, meta


def _save_pt_from_memmap(ids_path: str, attn_path: str, lab_path: str, meta: dict, pt_output_path: str):
    print(f"[pt] 从 memmap 生成: {pt_output_path}")
    # 优先以 r+ 打开，确保 numpy 数组 writeable，从而避免 torch.from_numpy 的只读告警；
    # 若失败则回退为只读并在转换时复制（内存会有一次性占用）。
    def _load_rw_or_ro(p: str):
        try:
            arr = np.load(p, mmap_mode='r+')
            return arr, True
        except Exception:
            try:
                arr = np.load(p, mmap_mode='r')
                return arr, False
            except Exception:
                # 最后的兜底：非内存映射加载
                arr = np.load(p)
                return arr, arr.flags.writeable

    ids_np, ids_rw = _load_rw_or_ro(ids_path)
    attn_np, attn_rw = _load_rw_or_ro(attn_path)
    lab_np, lab_rw = _load_rw_or_ro(lab_path)

    def _to_tensor(arr, rw: bool):
        if rw and getattr(arr, 'flags', None) is not None and arr.flags.writeable:
            return torch.from_numpy(arr)
        # 回退：复制生成张量，避免只读告警
        return torch.tensor(arr)

    bundle = {
        'input_ids': _to_tensor(ids_np, ids_rw),
        'attention_mask': _to_tensor(attn_np, attn_rw),
        'labels': _to_tensor(lab_np, lab_rw),
        'meta': {
            'num_samples': int(meta['num_samples']),
            'max_seq_length': int(meta['max_seq_length']),
            'avg_seq_length': None,
            'pad_token_id': int(meta['pad_token_id'])
        }
    }
    torch.save(bundle, pt_output_path)
    print(f"[pt] 保存完成: {pt_output_path}")


def process(input_path: str, output_path: str, max_seq_length: int | None, pad_token_id: int, overwrite: bool, enable_memmap: bool, stream: bool, save_pt: bool):
    if (not overwrite) and os.path.exists(output_path):
        print(f"跳过已存在: {output_path}")
        return
    print(f"读取 {input_path}")

    if stream:
        base = output_path[:-3]
        # 若未启用memmap, 强制启用（流式主要产出memmap）
        if not enable_memmap:
            print("[警告] 流式模式未启用 memmap, 将自动启用。")
            enable_memmap = True
        # 优先使用 metadata.json 中的信息，跳过首遍扫描
        N = None
        L = None
        try:
            meta_path = os.path.join(os.path.dirname(input_path), 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_json = json.load(f)
                # 推断该输入属于哪个 split
                split_paths = (meta_json.get('split_paths') or {})
                splits_info = (meta_json.get('splits') or {})
                which = None
                for name, p in split_paths.items():
                    try:
                        if os.path.abspath(p) == os.path.abspath(input_path):
                            which = name
                            break
                    except Exception:
                        continue
                if which and isinstance(splits_info.get(which), (int, float)):
                    N = int(splits_info[which])
                # L 优先使用 pack_config.max_seq_length，否则使用 preprocess.metadata 中的观测最大长度
                tok = meta_json.get('tokenize') or {}
                observed_max = tok.get('observed_max_input_len')
                if max_seq_length is not None:
                    L = int(max_seq_length)
                elif isinstance(observed_max, (int, float)):
                    L = int(observed_max)
        except Exception as e:
            print(f"[metadata] 读取失败，将退出: {e}")
            return
        # 若缺失信息则回退扫描
        if not isinstance(N, int) or N <= 0 or not isinstance(L, int) or L <= 0:
            # 第一遍扫描得到 N 和 L
            N, L = _scan_jsonl(input_path, max_seq_length)
        if N == 0:
            print("空数据，跳过")
            return
        # 第二遍写入 memmap
        ids_path, attn_path, lab_path, meta = _stream_pack_to_memmap(input_path, base, N, L, pad_token_id)
        # 可选从 memmap 生成 .pt（不会将全部载入内存）
        if save_pt:
            _save_pt_from_memmap(ids_path, attn_path, lab_path, meta, output_path)
        else:
            print("[pt] 已跳过 .pt 生成 (pack_config.pt=false)")
    else:
        samples = read_jsonl(input_path)
        print(f"样本数 {len(samples)}  max_seq_length={max_seq_length or 'auto'}")
        bundle = pack(samples, max_seq_length, pad_token_id)
        torch.save(bundle, output_path)
        print(f"保存 {output_path}  meta={bundle['meta']}")
        if enable_memmap:
            base = output_path[:-3]
            meta = bundle['meta']
            N = meta['num_samples']
            L = meta['max_seq_length']
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
                print(f"[memmap] 跳过已存在 memmap 文件 (覆盖需开启 overwrite)")

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
    stream = bool(pack_cfg.get('stream', False))
    save_pt = bool(pack_cfg.get('pt', True))

    # 轻量命令行覆盖，便于临时执行
    argv = set(sys.argv[1:])
    if '--stream' in argv:
        stream = True
    if '--memmap' in argv:
        enable_memmap = True
    if '--no-pt' in argv:
        save_pt = False
    if '--overwrite' in argv:
        overwrite = True

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
    print(f'suffix={suffix} pad_token_id={pad_id} max_seq_length={max_seq_len or "auto"} overwrite={overwrite} memmap={enable_memmap} stream={stream} pt={save_pt}')
    for inp, outp in tqdm(candidates, desc='文件打包', unit='文件'):
        process(inp, outp, max_seq_len, pad_id, overwrite, enable_memmap, stream, save_pt)
    print('=== 打包完成 ===')

if __name__ == '__main__':
    main()
