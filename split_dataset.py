import json, os, sys, math, random, collections, argparse, threading, gc
from queue import Queue
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


def _writer_worker(path: str, q: Queue):
    """后台写线程：从队列读取已带换行的字符串并写入文件，遇到 None 结束。"""
    with open(path, "w", encoding="utf-8") as f:
        while True:
            item = q.get()
            if item is None:
                break
            f.write(item)


def _label_to_int(v) -> int:
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        iv = int(v)
    except Exception:
        iv = 0
    return 1 if iv == 1 else 0


def stream_split_balance_binary(
    input_path: str,
    outdir: str,
    ratios: Tuple[float, float, float],
    seed: int,
    label_key: str,
    limit: int | None,
    meta_counts: Dict[str, int] | None = None,
    queue_size: int = 1000,
):
    """两遍扫描的流式平衡+分层划分：
    1) 遍历统计总行数与二分类计数。
    2) 以每类 min_count 为样本数做蓄水池采样，得到平衡集合；对每类样本打乱并按比例切分；
       通过三个写线程异步写出 train/val/test JSONL，并展示进度条。
    """
    if not input_path.endswith(".jsonl"):
        print("错误: 流式模式要求输入为 JSONL")
        sys.exit(1)

    # 如果提供了 metadata 计数，则跳过第一遍扫描
    if meta_counts is not None:
        total_lines = int(meta_counts.get("total_records", 0))
        n1 = int(meta_counts.get("label_1", 0))
        n0 = int(meta_counts.get("label_0", max(0, total_lines - n1)))
    else:
        print("无效的 metadata，退出")
        return 0, 0, 0
    if n0 + n1 == 0:
        print("无有效样本，退出")
        return 0, 0, 0
    print(f"原始类别计数: class0={n0} class1={n1}  (正例比例={n1/(n0+n1):.3f})")
    min_count = min(n0, n1)
    if limit and isinstance(limit, int) and limit > 0:
        per_class_cap = max(1, limit // 2)
        if per_class_cap < min_count:
            print(f"应用样本上限: 每类 {min_count} -> {per_class_cap}")
        min_count = min(min_count, per_class_cap)
    if min_count == 0:
        print("某一类别计数为0，无法平衡，退出")
        return 0, 0, 0

    # 按类进行蓄水池采样
    print("[流式] 蓄水池采样平衡集合")
    rng = random.Random(seed)
    reservoir = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    with open(input_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_lines, desc="采样", unit="行")
        for i_line, line in enumerate(f):
            pbar.update(1)
            try:
                r = json.loads(line)
            except Exception:
                continue
            cls = _label_to_int(r.get(label_key, 0))
            seen[cls] += 1
            res = reservoir[cls]
            k = len(res)
            if k < min_count:
                res.append(r)
            else:
                # 算法 R: 以 min_count/seen[cls] 的概率替换一个位置
                j = rng.randint(0, seen[cls] - 1)
                if j < min_count:
                    res[j] = r
            # 周期触发一次垃圾回收
            if (i_line % 200000) == 0:
                gc.collect()
        pbar.close()
    # 采样阶段结束，做一次显式回收
    gc.collect()

    # 分层按比例切分
    def split_one(lst: List[Dict[str, Any]]):
        rng.shuffle(lst)
        total = len(lst)
        tr_n = int(round(total * ratios[0]))
        val_n = int(round(total * ratios[1]))
        test_n = total - tr_n - val_n
        return lst[:tr_n], lst[tr_n : tr_n + val_n], lst[tr_n + val_n :]

    t0, v0, te0 = split_one(reservoir[0])
    t1, v1, te1 = split_one(reservoir[1])
    # 不再合并成大列表，避免额外内存；分别就地乱序后分阶段写出
    rng.shuffle(t0); rng.shuffle(t1)
    rng.shuffle(v0); rng.shuffle(v1)
    rng.shuffle(te0); rng.shuffle(te1)

    # 异步写入
    os.makedirs(outdir, exist_ok=True)
    train_path = os.path.join(outdir, "train.jsonl")
    val_path = os.path.join(outdir, "val.jsonl")
    test_path = os.path.join(outdir, "test.jsonl")
    tq, vq, eq = Queue(maxsize=max(10, queue_size)), Queue(maxsize=max(10, queue_size)), Queue(maxsize=max(10, queue_size))
    tt = threading.Thread(target=_writer_worker, args=(train_path, tq), daemon=True)
    vt = threading.Thread(target=_writer_worker, args=(val_path, vq), daemon=True)
    et = threading.Thread(target=_writer_worker, args=(test_path, eq), daemon=True)
    tt.start()
    vt.start()
    et.start()

    total_write = len(t0) + len(t1) + len(v0) + len(v1) + len(te0) + len(te1)
    pbar_w = tqdm(total=total_write, desc="写出", unit="条")
    # 分阶段写出，且在阶段间回收内存
    for r in t0:
        tq.put(json.dumps(r, ensure_ascii=False) + "\n")
        pbar_w.update(1)
    t0.clear(); gc.collect()
    for r in t1:
        tq.put(json.dumps(r, ensure_ascii=False) + "\n")
        pbar_w.update(1)
    t1.clear(); gc.collect()
    for r in v0:
        vq.put(json.dumps(r, ensure_ascii=False) + "\n")
        pbar_w.update(1)
    v0.clear(); gc.collect()
    for r in v1:
        vq.put(json.dumps(r, ensure_ascii=False) + "\n")
        pbar_w.update(1)
    v1.clear(); gc.collect()
    for r in te0:
        eq.put(json.dumps(r, ensure_ascii=False) + "\n")
        pbar_w.update(1)
    te0.clear(); gc.collect()
    for r in te1:
        eq.put(json.dumps(r, ensure_ascii=False) + "\n")
        pbar_w.update(1)
    te1.clear(); gc.collect()
    pbar_w.close()

    # 结束与同步
    for q in (tq, vq, eq):
        q.put(None)
    for t in (tt, vt, et):
        t.join()

    # 返回各 split 样本数（与写出一致）
    return (len(t0) + len(t1)), (len(v0) + len(v1)), (len(te0) + len(te1))


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


def main():
    parser = argparse.ArgumentParser(description="数据集划分工具 (支持流式)")
    parser.add_argument(
        "--stream", action="store_true", help="启用流式平衡+划分+写入，适用于大JSONL"
    )
    args, _ = parser.parse_known_args()
    config_file = DEFAULT_CONFIG_FILE
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在 {config_file}")
        sys.exit(1)
    with open(config_file, "r", encoding="utf-8") as f:
        full_cfg = json.load(f)
    split_cfg = full_cfg.get("split_config") or {}

    # 若存在 metadata.json，优先读取分词输出路径
    meta_path = os.path.join(DEFAULT_OUT_DIR, "metadata.json")
    input_path = split_cfg.get("input")
    if (not input_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cand = (meta.get("paths") or {}).get("tokenized_jsonl")
            if cand:
                input_path = cand
                print(f"使用 metadata 指定的输入: {input_path}")
        except Exception:
            input_path = None
    input_path = input_path or DEFAULT_INPUT
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

    # 预先初始化，便于后续写入 metadata 时引用
    tr_n = va_n = te_n = 0
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    if args.stream:
        print(f"\n--- 流式: 平衡并分层划分 (二分类) ---")
        # 若 metadata 提供 label 统计，直接跳过第一遍计数
        meta_counts = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_json = json.load(f)
                mc = meta_json.get("counts") or {}
                if isinstance(mc, dict) and (
                    "label_0" in mc or "label_1" in mc or "total_records" in mc
                ):
                    meta_counts = {
                        "total_records": int(mc.get("total_records", 0)),
                        "label_1": int(mc.get("label_1", 0)),
                        "label_0": int(mc.get("label_0", 0)),
                    }
            except Exception:
                meta_counts = None
        tr_n, va_n, te_n = stream_split_balance_binary(
            input_path, outdir, ratios_tuple, seed, label_key, limit, meta_counts
        )
    else:
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

    print(f"\n=== 划分完成 ===")
    # 非流式时已在分支内打印详细信息
    print(f"输出目录: {outdir}")

    # 将分割后的样本数与路径写入 metadata.json，供 pack 使用
    try:
        train_path = os.path.join(outdir, "train.jsonl")
        val_path = os.path.join(outdir, "val.jsonl")
        test_path = os.path.join(outdir, "test.jsonl")
        if args.stream:
            sizes = {
                "train": int(tr_n),
                "val": int(va_n),
                "test": int(te_n),
            }
        else:
            sizes = {
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
            }
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                try:
                    meta = json.load(f) or {}
                except Exception:
                    meta = {}
        meta.setdefault("splits", {})
        meta.setdefault("split_paths", {})
        meta["splits"].update(sizes)
        meta["split_paths"].update(
            {
                "train": train_path,
                "val": val_path,
                "test": test_path,
            }
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"已更新元数据: {meta_path}")
    except Exception as e:
        print(f"警告: 写入 metadata.json 失败: {e}")


if __name__ == "__main__":
    main()
