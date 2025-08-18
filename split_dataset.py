import json, os, sys, math, random, collections
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# 默认路径
DEFAULT_INPUT = os.path.join("dataset", "tokenized_data.jsonl")
DEFAULT_OUT_DIR = "dataset"
DEFAULT_CONFIG_FILE = os.path.join("config", "config.json")


def load_data(path: str) -> List[Dict[str, Any]]:
    """
    将数据文件（json或jsonl）完全读入内存
    """
    print(f"正在读取数据文件: {path}")

    # 检查文件是否存在
    if not os.path.exists(path):
        print(f"错误: 文件不存在 {path}")
        sys.exit(1)

    # 获取文件大小信息
    file_size = os.path.getsize(path)
    print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")

    # 读取数据到内存
    records = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            print("正在解析JSONL数据...")
            for line in tqdm(f, desc=f"读取 {os.path.basename(path)}", unit="行"):
                records.append(json.loads(line))
        else:
            print("正在解析JSON数据...")
            records = json.load(f)

    print(f"数据已读入内存，共 {len(records)} 条记录")
    return records


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    """
    将记录列表写入JSONL文件，显示进度条
    """
    print(f"正在写入 {os.path.basename(path)} ({len(records)} 条记录)")
    with open(path, "w", encoding="utf-8") as f:
        for r in tqdm(records, desc=f"写入 {os.path.basename(path)}", unit="条"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成写入: {path}")


def stratified_split_balance_binary(
    records: List[Dict[str, Any]],
    ratios: Tuple[float, float, float],
    seed: int,
    label_key: str,
):
    """对二分类数据进行类别平衡(下采样多数类) + 分层划分。

    步骤:
      1. 将记录按 label_key (0/1 或 False/True) 分组。
      2. 取两组的最小数量 min_count, 随机采样各 min_count 条形成平衡集合。
      3. 对平衡集合按类别分别 shuffle 后按比例划分 (保证各 split 类别数量尽可能接近)。
      4. 合并两个类别对应 split, 再整体 shuffle (保证交错)。
    """
    rng = random.Random(seed)
    group = {0: [], 1: []}
    for r in records:
        v = r.get(label_key, 0)
        if isinstance(v, bool):
            v = 1 if v else 0
        try:
            v = int(v)
        except Exception:
            v = 0
        if v not in (0, 1):
            v = 0
        group[v].append(r)

    n0, n1 = len(group[0]), len(group[1])

    print(f"原始类别计数: class0={n0} class1={n1}  (ratio={n1/(n0+n1):.3f} 正例比例)")
    min_count = min(n0, n1)
    print(f"按较少类别数 {min_count} 下采样另一个类别以实现平衡。")
    sampled0 = rng.sample(group[0], min_count) if n0 > min_count else group[0]
    sampled1 = rng.sample(group[1], min_count) if n1 > min_count else group[1]

    # 分层按比例划分
    def split_one(lst: List[Dict[str, Any]]):
        rng.shuffle(lst)
        total = len(lst)
        tr_n = int(round(total * ratios[0]))
        val_n = int(round(total * ratios[1]))
        # 余数给 test
        test_n = total - tr_n - val_n
        train_part = lst[:tr_n]
        val_part = lst[tr_n : tr_n + val_n]
        test_part = lst[tr_n + val_n :]
        return train_part, val_part, test_part

    t0, v0, te0 = split_one(sampled0)
    t1, v1, te1 = split_one(sampled1)

    train = t0 + t1
    val = v0 + v1
    test = te0 + te1
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    def stats(name, part):
        c0 = sum(1 for r in part if int(r.get(label_key, 0)) == 0)
        c1 = sum(1 for r in part if int(r.get(label_key, 0)) == 1)
        tot = len(part) or 1
        print(
            f"  {name}: {len(part)} 条  class0={c0} class1={c1}  正例比例={c1/tot:.3f}"
        )

    print("[平衡+分层] 划分结果:")
    stats("train", train)
    stats("val", val)
    stats("test", test)
    return train, val, test


def compute_stats(
    name: str, data: List[Dict[str, Any]], label_key: str
) -> Dict[str, Any]:
    """
    计算数据集统计信息，显示计算进度
    """
    print(f"正在计算 {name} 集统计信息...")

    # 在内存中计算标签频率
    freq = collections.Counter()
    for r in tqdm(data, desc=f"统计 {name} 标签", unit="条", leave=False):
        freq[r.get(label_key, "UNKNOWN")] += 1

    if not data:
        entropy = 0.0
    else:
        import math

        total = len(data)
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())

    stats = {
        "split": name,
        "size": len(data),
        "unique_labels": len(freq),
        "top5": freq.most_common(5),
        "entropy": round(entropy, 4),
    }

    print(
        f"{name} 集统计: {len(data)} 条记录, {len(freq)} 个唯一标签, 熵={entropy:.4f}"
    )
    return stats


def main():
    config_file = DEFAULT_CONFIG_FILE
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在 {config_file}")
        sys.exit(1)
    with open(config_file, "r", encoding="utf-8") as f:
        full_cfg = json.load(f)
    split_cfg = full_cfg.get("splitConfig") or {}

    input_path = split_cfg.get("input", DEFAULT_INPUT)
    outdir = split_cfg.get("outdir", DEFAULT_OUT_DIR)
    label_key = split_cfg.get("label_key", "label")
    ratios_raw = split_cfg.get("ratios", "0.8,0.1,0.1")
    seed = split_cfg.get("seed", 42)
    limit = split_cfg.get("limit")  # 可选: 限制抽样数量

    print("=== 数据集划分工具 ===")
    print(f"配置文件: {config_file}")
    print(f"输入文件: {input_path}")
    print(f"输出目录: {outdir}")
    print(f"标签键: {label_key}")
    print(f"划分比例: {ratios_raw}")
    print(f"随机种子: {seed}")
    if limit:
        print(f"样本限制: 取前/随机抽样 {limit} 条 (若可用)")

    ratios_tuple = tuple(
        float(x)
        for x in (
            ratios_raw
            if isinstance(ratios_raw, str)
            else ",".join(map(str, ratios_raw))
        ).split(",")
    )
    if len(ratios_tuple) != 3 or not math.isclose(sum(ratios_tuple), 1.0, rel_tol=1e-3):
        print("错误: 比例必须是3个数字且和为1.0")
        sys.exit(1)

    print(f"\n--- 第1步: 读取数据 ---")
    data = load_data(input_path)
    if not data:
        print("数据为空，退出")
        sys.exit(0)

    original_total = len(data)
    if limit and isinstance(limit, int) and 0 < limit < original_total:
        rng = random.Random(seed)
        print(f"应用样本限制: 原始 {original_total} -> 抽样 {limit}")
        data = rng.sample(data, limit)
        print(f"抽样完成: 当前样本数 {len(data)}")

    print(f"\n--- 第2步: 平衡并分层划分 (二分类) ---")
    train, val, test = stratified_split_balance_binary(
        data, ratios_tuple, seed, label_key
    )

    print(f"\n--- 第3步: 写入文件 ---")
    os.makedirs(outdir, exist_ok=True)
    files_to_write = [
        (os.path.join(outdir, "train.jsonl"), train),
        (os.path.join(outdir, "val.jsonl"), val),
        (os.path.join(outdir, "test.jsonl"), test),
    ]
    for filepath, dataset in files_to_write:
        write_jsonl(filepath, dataset)

    print(f"\n--- 第4步: 计算统计信息 ---")
    stats = [
        compute_stats("train", train, label_key),
        compute_stats("val", val, label_key),
        compute_stats("test", test, label_key),
        {
            "total": len(data),
            "seed": seed,
            "ratios": list(ratios_tuple),
            "original_total": original_total,
            "limit": int(limit) if limit else None,
        },
    ]
    stats_file = os.path.join(outdir, "split_stats.json")
    print(f"正在保存统计信息到: {stats_file}")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n=== 划分完成 ===")
    print(f"训练集: {len(train)} 条")
    print(f"验证集: {len(val)} 条")
    print(f"测试集: {len(test)} 条")
    print(f"总计: {len(data)} 条")
    print(f"输出目录: {outdir}")
    print(f"统计文件: {stats_file}")


if __name__ == "__main__":
    main()
