import json, os, sys, math, random, argparse, collections
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# 默认路径
DEFAULT_INPUT = os.path.join('dataset', 'tokenized_data.jsonl')
DEFAULT_OUT_DIR = 'dataset'
DEFAULT_CONFIG_FILE = os.path.join('config', 'config.json')


def load_data(path: str) -> List[Dict[str, Any]]:
    """
    将数据文件（json或jsonl）完全读入内存
    """
    print(f'正在读取数据文件: {path}')
    
    # 检查文件是否存在
    if not os.path.exists(path):
        print(f'错误: 文件不存在 {path}')
        sys.exit(1)
    
    # 获取文件大小信息
    file_size = os.path.getsize(path)
    print(f'文件大小: {file_size / 1024 / 1024:.2f} MB')
    
    # 读取数据到内存
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.jsonl'):
            print('正在解析JSONL数据...')
            for line in tqdm(f, desc=f"读取 {os.path.basename(path)}", unit='行'):
                records.append(json.loads(line))
        else:
            print('正在解析JSON数据...')
            records = json.load(f)
    
    print(f'数据已读入内存，共 {len(records)} 条记录')
    return records


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    """
    将记录列表写入JSONL文件，显示进度条
    """
    print(f'正在写入 {os.path.basename(path)} ({len(records)} 条记录)')
    with open(path, 'w', encoding='utf-8') as f:
        for r in tqdm(records, desc=f'写入 {os.path.basename(path)}', unit='条'):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'完成写入: {path}')


def global_random_split(records: List[Dict[str, Any]], ratios: Tuple[float, float, float], seed: int):
    """
    在内存中进行全局随机划分，显示进度
    """
    train_ratio, val_ratio, test_ratio = ratios
    print(f'\n--- 开始数据划分 ---')
    print(f'划分比例: 训练集 {train_ratio:.1%}, 验证集 {val_ratio:.1%}, 测试集 {test_ratio:.1%}')
    print(f'随机种子: {seed}')
    
    # 在内存中打乱数据
    print('正在打乱数据顺序...')
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    
    # 计算划分点
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    
    print(f'数据划分计划:')
    print(f'  训练集: {n_train} 条 ({n_train/n:.1%})')
    print(f'  验证集: {n_val} 条 ({n_val/n:.1%})')
    print(f'  测试集: {n_test} 条 ({n_test/n:.1%})')
    
    # 在内存中进行划分
    print('正在划分数据集...')
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    
    print('数据划分完成')
    return train, val, test


def compute_stats(name: str, data: List[Dict[str, Any]], label_key: str) -> Dict[str, Any]:
    """
    计算数据集统计信息，显示计算进度
    """
    print(f'正在计算 {name} 集统计信息...')
    
    # 在内存中计算标签频率
    freq = collections.Counter()
    for r in tqdm(data, desc=f'统计 {name} 标签', unit='条', leave=False):
        freq[r.get(label_key, 'UNKNOWN')] += 1
    
    if not data:
        entropy = 0.0
    else:
        import math
        total = len(data)
        entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())
    
    stats = {
        'split': name,
        'size': len(data),
        'unique_labels': len(freq),
        'top5': freq.most_common(5),
        'entropy': round(entropy, 4)
    }
    
    print(f'{name} 集统计: {len(data)} 条记录, {len(freq)} 个唯一标签, 熵={entropy:.4f}')
    return stats


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test (支持统一 config.json 的 splitConfig)')
    parser.add_argument('--config_file', default=DEFAULT_CONFIG_FILE, help='配置文件 (包含 splitConfig) 路径')
    parser.add_argument('--input', default=None, help='输入 data.json 路径 (若提供则覆盖配置)')
    parser.add_argument('--outdir', default=None, help='输出目录 (若提供则覆盖配置)')
    parser.add_argument('--label-key', default=None, help='主标签键名 (若提供则覆盖配置)')
    parser.add_argument('--ratios', default=None, help='train,val,test 比例 (若提供则覆盖配置)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子 (若提供则覆盖配置)')
    args = parser.parse_args()

    # 读取配置文件
    if not os.path.exists(args.config_file):
        print(f'错误: 配置文件不存在 {args.config_file}')
        sys.exit(1)
    with open(args.config_file, 'r', encoding='utf-8') as f:
        full_cfg = json.load(f)
    split_cfg = full_cfg.get('splitConfig') or {}

    # 从 splitConfig 读取，CLI 非空则覆盖
    input_path = args.input or split_cfg.get('input', DEFAULT_INPUT)
    outdir = args.outdir or split_cfg.get('outdir', DEFAULT_OUT_DIR)
    label_key = args.label_key or split_cfg.get('label_key', 'label')
    ratios_raw = args.ratios or split_cfg.get('ratios', '0.8,0.1,0.1')
    seed = args.seed if args.seed is not None else split_cfg.get('seed', 42)

    print('=== 数据集划分工具 ===')
    print(f'配置文件: {args.config_file}')
    print(f'输入文件: {input_path}')
    print(f'输出目录: {outdir}')
    print(f'标签键: {label_key}')
    print(f'划分比例: {ratios_raw}')
    print(f'随机种子: {seed}')

    # 解析划分比例
    ratios_tuple = tuple(float(x) for x in (ratios_raw if isinstance(ratios_raw, str) else ','.join(map(str, ratios_raw))).split(','))
    if len(ratios_tuple) != 3 or not math.isclose(sum(ratios_tuple), 1.0, rel_tol=1e-3):
        print('错误: 比例必须是3个数字且和为1.0')
        sys.exit(1)

    # 第一步：读取数据到内存
    print(f'\n--- 第1步: 读取数据 ---')
    data = load_data(input_path)
    if not data:
        print('数据为空，退出')
        sys.exit(0)

    # 第二步：在内存中进行数据划分
    print(f'\n--- 第2步: 划分数据 ---')
    train, val, test = global_random_split(data, ratios_tuple, args.seed)

    # 第三步：创建输出目录并写入文件
    print(f'\n--- 第3步: 写入文件 ---')
    os.makedirs(outdir, exist_ok=True)
    
    # 并行写入三个文件
    files_to_write = [
    (os.path.join(outdir, 'train.jsonl'), train),
    (os.path.join(outdir, 'val.jsonl'), val),
    (os.path.join(outdir, 'test.jsonl'), test)
    ]
    
    for filepath, dataset in files_to_write:
        write_jsonl(filepath, dataset)

    # 第四步：计算并保存统计信息
    print(f'\n--- 第4步: 计算统计信息 ---')
    stats = [
        compute_stats('train', train, args.label_key),
        compute_stats('val', val, args.label_key),
        compute_stats('test', test, args.label_key),
        {
            'total': len(data),
            'mode': 'global_random',
            'seed': args.seed,
            'ratios': list(ratios_tuple)
        }
    ]
    
    stats_file = os.path.join(outdir, 'split_stats.json')
    print(f'正在保存统计信息到: {stats_file}')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 第五步：显示完成信息
    print(f'\n=== 划分完成 ===')
    print(f'训练集: {len(train)} 条')
    print(f'验证集: {len(val)} 条')
    print(f'测试集: {len(test)} 条')
    print(f'总计: {len(data)} 条')
    print(f'输出目录: {outdir}')
    print(f'统计文件: {stats_file}')


if __name__ == '__main__':
    main()
