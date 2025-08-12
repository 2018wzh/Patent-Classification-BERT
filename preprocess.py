import csv
import json
import sys
import os
from typing import Dict, Any, List, Set

DEFAULT_CONFIG_PATH = os.path.join('config', 'config.json')


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg


def normalize_single_ipc(code: str) -> str:
    if not code:
        return ''
    code = code.strip().upper()
    # 替换全角与奇异字符
    code = code.replace('／', '/').replace('－', '-').replace('–', '-')
    # 去除内部多余空格
    code = code.replace(' ', '')
    # 去掉尾部的点
    code = code.rstrip('.')
    return code


def normalize_ipc_list(ipc: str) -> List[str]:
    if not ipc:
        return []
    # 分隔符可能包含 ; 、空格、, 、中文分号
    raw_parts = [p.strip() for p in ipc.replace('；', ';').replace(',', ';').split(';')]
    normed: List[str] = []
    seen: Set[str] = set()
    for p in raw_parts:
        np = normalize_single_ipc(p)
        if np and np not in seen:
            seen.add(np)
            normed.append(np)
    return normed


def derive_label(ipc_main: str) -> str:
    # 使用规范化后的主分类号，取前4个结构位 (字母 + 两位数字 + 字母) 作为标签
    if not ipc_main:
        return 'UNKNOWN'
    ipc_main = normalize_single_ipc(ipc_main)
    if not ipc_main:
        return 'UNKNOWN'
    # 典型格式: H01M4/24 或 A61K31/00
    # 截断到第一个 '/' 之前
    before_slash = ipc_main.split('/')[0]
    # 至少前4位（若不足直接返回全部）
    if len(before_slash) >= 4:
        return before_slash[:4]
    return before_slash


def unify_record(row: Dict[str, str], field_mapping: Dict[str, str], label_field_cn: str) -> Dict[str, Any]:
    # 标准化 IPC
    all_ipc_raw = row.get('IPC分类号', '')
    main_ipc_raw = row.get('IPC主分类号', '')
    all_ipc_list = normalize_ipc_list(all_ipc_raw)
    main_ipc_norm = normalize_single_ipc(main_ipc_raw)

    # 主标签
    primary_label = derive_label(main_ipc_norm)
    # 所有 IPC 转换为标签集合
    label_set: Set[str] = set()
    for code in all_ipc_list:
        lbl = derive_label(code)
        if lbl:
            label_set.add(lbl)
    if primary_label:
        label_set.add(primary_label)
    # 去除 UNKNOWN 以外空值，如只有 UNKNOWN 则保留
    if 'UNKNOWN' in label_set and len(label_set) > 1:
        label_set.discard('UNKNOWN')
    labels = sorted(label_set)

    parts = [
        row.get('专利名称', ''),
        row.get('摘要文本', ''),
        row.get('主权项内容', ''),
    ]
    text = '\n'.join([p for p in parts if p])

    grant_no = row.get('授权公告号', '').strip()
    return {
        'id': grant_no,
        'text': text,
        'labels': all_ipc_list,
        'label': main_ipc_norm
    }


def process_csv_file(path: str, field_mapping: Dict[str, str], label_field_cn: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(unify_record(row, field_mapping, label_field_cn))
    return data


def main():
    # 允许: python preprocess.py 或 python preprocess.py <config_path>
    if len(sys.argv) > 2:
        print('用法: python preprocess.py [config_path]')
        sys.exit(1)
    config_path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f'配置文件不存在: {config_path}')
        sys.exit(1)

    cfg = load_config(config_path)
    convert_files = cfg.get('convertFiles', [])
    output_file = cfg.get('outputFile')
    if not output_file:
        print('config 缺少 outputFile')
        sys.exit(1)
    field_mapping = cfg.get('fieldMapping')
    if not field_mapping:
        print('config 缺少 fieldMapping')
        sys.exit(1)
    label_field_cn = cfg.get('labelField', 'IPC主分类号')

    all_records: List[Dict[str, Any]] = []
    for rel in convert_files:
        csv_path = rel if os.path.isabs(rel) else os.path.join(os.path.dirname(config_path), '..', rel).replace('..'+os.sep+os.sep, '..'+os.sep)
        if not os.path.exists(csv_path):
            csv_path = rel if os.path.isabs(rel) else os.path.join(os.getcwd(), rel)
        if not os.path.exists(csv_path):
            print(f'警告: 找不到文件 {rel}, 已跳过')
            continue
        print(f'处理: {csv_path}')
        records = process_csv_file(csv_path, field_mapping, label_field_cn)
        all_records.extend(records)

    # 仅使用配置的 validLabels 进行前缀模糊匹配
    cfg_valid = cfg.get('validLabels') or []
    cfg_valid_norm = sorted(set(normalize_single_ipc(v) for v in cfg_valid if v), key=lambda x: (-len(x), x))

    def fuzzy_valid(codes: List[str]) -> bool:
        for c in codes:
            nc = normalize_single_ipc(c)
            for pref in cfg_valid_norm:
                if nc.startswith(pref):
                    return True
        return False

    for r in all_records:
        val = fuzzy_valid(r.get('labels', []))
        r['valid'] = val
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f'完成: {len(all_records)} 条记录 -> {output_file} (含 valid 字段, 不输出独立 valid_labels.json)')


if __name__ == '__main__':
    main()
