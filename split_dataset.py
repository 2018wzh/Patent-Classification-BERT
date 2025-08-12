import json, os, sys, math, random, argparse, collections
from typing import List, Dict, Any, Tuple

# 默认路径
DEFAULT_INPUT = os.path.join('dataset', 'data.json')
DEFAULT_OUT_DIR = 'dataset'


def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def global_random_split(records: List[Dict[str, Any]], ratios: Tuple[float, float, float], seed: int):
    train_ratio, val_ratio, test_ratio = ratios
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def compute_stats(name: str, data: List[Dict[str, Any]], label_key: str) -> Dict[str, Any]:
    freq = collections.Counter(r.get(label_key, 'UNKNOWN') for r in data)
    if not data:
        entropy = 0.0
    else:
        import math
        total = len(data)
        entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())
    return {
        'split': name,
        'size': len(data),
        'unique_labels': len(freq),
        'top5': freq.most_common(5),
        'entropy': round(entropy, 4)
    }


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='输入 data.json 路径')
    parser.add_argument('--outdir', default=DEFAULT_OUT_DIR, help='输出目录')
    parser.add_argument('--label-key', default='label', help='主标签键名')
    parser.add_argument('--ratios', default='0.8,0.1,0.1', help='train,val,test 比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    ratios_tuple = tuple(float(x) for x in args.ratios.split(','))
    if len(ratios_tuple) != 3 or not math.isclose(sum(ratios_tuple), 1.0, rel_tol=1e-3):
        print('比例必须 3 个且和为 1.0')
        sys.exit(1)

    data = load_data(args.input)
    if not data:
        print('数据为空，退出')
        sys.exit(0)

    train, val, test = global_random_split(data, ratios_tuple, args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    write_jsonl(os.path.join(args.outdir, 'train.jsonl'), train)
    write_jsonl(os.path.join(args.outdir, 'val.jsonl'), val)
    write_jsonl(os.path.join(args.outdir, 'test.jsonl'), test)

    # 保存统计信息
    # 生成 label2id (按频次降序)
    label_freq_total = collections.Counter(r.get(args.label_key, 'UNKNOWN') for r in data)
    sorted_labels = [lbl for lbl, _ in label_freq_total.most_common()]
    label2id = {lbl: idx for idx, lbl in enumerate(sorted_labels)}
    with open(os.path.join(args.outdir, 'label2id.json'), 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    stats = [
        compute_stats('train', train, args.label_key),
        compute_stats('val', val, args.label_key),
        compute_stats('test', test, args.label_key),
        {
            'total': len(data),
            'mode': 'global_random'
        }
    ]
    with open(os.path.join(args.outdir, 'split_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('完成划分: train', len(train), 'val', len(val), 'test', len(test))


if __name__ == '__main__':
    main()
