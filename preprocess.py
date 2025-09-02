import csv
import json
import sys
import os
import argparse
import glob
import math
import threading
from queue import Queue
from typing import Dict, Any, List, Set, Optional, Tuple
from tqdm import tqdm
from transformers import BertTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_CONFIG_PATH = os.path.join("config", "config.json")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def tokenize_and_format(
    records: List[Dict[str, Any]], train_config: Dict[str, Any], preprocess_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    model_name = train_config.get("model")
    if not model_name:
        print("错误: 训练配置中缺少 'model' 名称。")
        sys.exit(1)

    max_seq_length = train_config.get("max_seq_length", 512)
    text_column = train_config.get("text_column_name", "text")
    label_column = train_config.get("label_column_name", "label")
    batch_size = preprocess_config.get("batch_size", 32)
    workers = preprocess_config.get("workers", 1)
    print("\n--- 开始批量分词 (fast tokenizer, CPU) ---")
    print(
        f"模型/分词器: {model_name}  最大序列长度: {max_seq_length}  批次: {batch_size}  线程: {workers}"
    )
    tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=True)

    texts = [rec[text_column] for rec in records]
    labels = [1 if rec.get(label_column) else 0 for rec in records]
    original_ids = [rec.get("id", "") for rec in records]
    total = len(texts)
    out: List[Dict[str, Any]] = []

    if workers == 1:
        # 单线程: 使用样本级进度条
        pbar = tqdm(total=total, desc="分词", unit="样本")
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]
            enc = tokenizer(
                batch_texts,
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
            )
            for i, input_ids in enumerate(enc["input_ids"]):
                out.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": enc["attention_mask"][i],
                        "label": labels[start + i],
                        "original_id": original_ids[start + i],
                    }
                )
            pbar.update(end - start)
        pbar.close()
    else:
        # 多线程：以 batch 为单位并行，将结果按 start 排序后合并
        batch_starts = list(range(0, total, batch_size))
        futures = []
        def encode_range(start: int) -> Tuple[int, Dict[str, List[List[int]]]]:
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]
            enc_local = tokenizer(
                batch_texts,
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
            )
            return start, enc_local
        results: List[Tuple[int, Dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for s in batch_starts:
                futures.append(ex.submit(encode_range, s))
            pbar = tqdm(total=total, desc="分词(多线程)", unit="样本")
            for fut in as_completed(futures):
                start, enc = fut.result()
                results.append((start, enc))
                end = min(start + batch_size, total)
                pbar.update(end - start)
            pbar.close()
        # 按 start 排序，保证稳定顺序
        results.sort(key=lambda x: x[0])
        for start, enc in results:
            for i, input_ids in enumerate(enc["input_ids"]):
                out.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": enc["attention_mask"][i],
                        "label": labels[start + i],
                        "original_id": original_ids[start + i],
                    }
                )
    print(f"分词完成，总样本: {len(out)}")
    return out

    # 全角转半角（字母和数字）


def to_halfwidth(s: str) -> str:
    res = []
    for c in s:
        codepoint = ord(c)
        # 全角数字
        if 0xFF10 <= codepoint <= 0xFF19:
            res.append(chr(codepoint - 0xFF10 + ord("0")))
        # 全角大写字母
        elif 0xFF21 <= codepoint <= 0xFF3A:
            res.append(chr(codepoint - 0xFF21 + ord("A")))
        # 全角小写字母
        elif 0xFF41 <= codepoint <= 0xFF5A:
            res.append(chr(codepoint - 0xFF41 + ord("a")))
        # 全角斜杠、减号等
        elif c == "／":
            res.append("/")
        elif c in {"－", "–"}:
            res.append("-")
        else:
            res.append(c)
    return "".join(res)


def normalize_single_ipc(code: str) -> str:
    if not code:
        return ""
    # 去除内部多余空格
    code = code.replace(" ", "")
    # 去掉尾部的点
    code = code.rstrip(".")
    return code


def normalize_ipc_list(ipc: str) -> List[str]:
    if not ipc:
        return []
    # 分隔符可能包含 ; 、空格、, 、中文分号
    raw_parts = [p.strip() for p in ipc.replace("；", ";").replace(",", ";").split(";")]
    normed: List[str] = []
    seen: Set[str] = set()
    for p in raw_parts:
        np = normalize_single_ipc(p)
        if np and np not in seen:
            seen.add(np)
            normed.append(np)
    return normed


def _normalize_row_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    """去除CSV行中键名的 BOM(\ufeff) 与首尾空白，避免首列丢失。"""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(k, str):
            nk = k.lstrip("\ufeff").strip()
        else:
            nk = k
        out[nk] = v
    return out


def _writer_worker(path: str, q: Queue):
    """后台写线程：从队列读取已带换行的字符串并写入文件，遇到 None 结束。"""
    # 使用缓冲写入，线程结束时自动关闭
    with open(path, "w", encoding="utf-8") as f:
        while True:
            item = q.get()
            if item is None:
                break
            f.write(item)
        # 文件关闭即刷新


def derive_label(ipc_main: str) -> str:
    # 使用规范化后的主分类号，取前4个结构位 (字母 + 两位数字 + 字母) 作为标签
    if not ipc_main:
        return "UNKNOWN"
    ipc_main = normalize_single_ipc(ipc_main)
    if not ipc_main:
        return "UNKNOWN"
    # 典型格式: H01M4/24 或 A61K31/00
    # 截断到第一个 '/' 之前
    before_slash = ipc_main.split("/")[0]
    # 至少前4位（若不足直接返回全部）
    if len(before_slash) >= 4:
        return before_slash[:4]
    return before_slash


def clean_text_content(text: str, remove_keywords: List[str]) -> str:
    """
    根据配置的关键字清洗文本内容
    """
    text = to_halfwidth(text)
    if not text or not remove_keywords:
        return text
    cleaned_text = text
    for keyword in remove_keywords:
        cleaned_text = cleaned_text.replace(keyword, "")

    # 清理多余的空白字符
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text


def unify_record(
    row: Dict[str, str], remove_keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    # 仅保留主分类号
    main_ipc_raw = row.get("IPC主分类号", "")
    main_ipc_norm = normalize_single_ipc(main_ipc_raw)

    # 获取文本内容并进行清洗
    patent_name = row.get("专利名称", "")
    abstract = row.get("摘要文本", "")
    claims = row.get("主权项内容", "")

    # 如果有配置的关键字，则进行清洗
    if remove_keywords:
        patent_name = clean_text_content(patent_name, remove_keywords)
        abstract = clean_text_content(abstract, remove_keywords)
        claims = clean_text_content(claims, remove_keywords)

    parts = [patent_name, abstract, claims]
    text = "\n".join([p for p in parts if p])

    grant_no = row.get("公开公告号", "").strip()
    return {
        "id": grant_no,
        "text": text,
        "ipc": main_ipc_norm,
        # 后续匹配写入 label 与 valid
    }


def process_csv_file(
    path: str, remove_keywords: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    将CSV文件完全读入内存后进行处理
    """
    print(f"正在读取文件到内存: {os.path.basename(path)}")

    # 第一步：将整个CSV文件读入内存
    all_rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = [_normalize_row_keys(r) for r in reader]

    print(f"文件已读入内存，共 {len(all_rows)} 行数据")

    # 第二步：在内存中处理所有行
    data: List[Dict[str, Any]] = []
    for row in tqdm(all_rows, desc=f"处理 {os.path.basename(path)}", unit="行"):
        processed_record = unify_record(row, remove_keywords)
        data.append(processed_record)

    return data


def process_csv_file_stream(
    path: str, remove_keywords: Optional[List[str]] = None
):
    """流式读取 CSV，每行生成一个统一结构的记录。"""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row = _normalize_row_keys(row)
                yield unify_record(row, remove_keywords)
            except Exception:
                continue


    


def write_cleaned_csv_one(
    in_path: str,
    out_path: str,
    selected_columns: List[str],
    remove_keywords: Optional[List[str]] = None,
):
    """对单个 CSV 进行清洗并输出到 out_path。返回统计信息。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total_read = 0
    total_written = 0
    desc = f"清洗: {os.path.basename(in_path)}"
    pbar = tqdm(total=None, desc=desc, unit="行")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(selected_columns)
        try:
            with open(in_path, "r", encoding="utf-8") as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    row = _normalize_row_keys(row)
                    total_read += 1
                    out_row: List[str] = []
                    for col in selected_columns:
                        val = row.get(col, "")
                        if isinstance(val, str):
                            val = clean_text_content(val, remove_keywords or [])
                        else:
                            val = "" if val is None else str(val)
                        out_row.append(val)
                    writer.writerow(out_row)
                    if pbar is not None:
                        pbar.update(1)
                    total_written += 1
        except Exception as e:
            print(f"警告: 清洗CSV处理失败 {in_path}: {e}")
        finally:
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass
    return {"read": total_read, "written": total_written}


def main():
    parser = argparse.ArgumentParser(
        description="预处理脚本: CSV -> 规范 JSON + tokenized jsonl"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径 (含 preprocess_config/train_config)",
    )
    parser.add_argument(
        "--convert-file",
        action="append",
        help="追加 / 覆盖要处理的 CSV 文件 (可多次指定)",
    )
    parser.add_argument("--model", help="覆盖 train_config.model")
    parser.add_argument(
        "--max-seq-length", type=int, help="覆盖 train_config.max_seq_length"
    )
    parser.add_argument(
        "--batch-size", type=int, help="覆盖 preprocess_config.batch_size"
    )
    parser.add_argument(
        "--workers", type=int, help="分词线程数 (覆盖 preprocess_config.workers, 默认1)"
    )
    parser.add_argument(
        "--remove-keyword", action="append", help="追加要删除的噪声关键字 (可多次)"
    )
    parser.add_argument(
        "--valid-label", action="append", help="追加有效标签前缀 (可多次)"
    )
    parser.add_argument(
        "--output-dir", default="dataset", help="输出目录 (默认 dataset)"
    )
    parser.add_argument(
        "--skip-tokenize", action="store_true", help="仅解析与标注 valid, 不进行分词"
    )
    parser.add_argument(
        "--stream", action="store_true", help="流式模式: 逐行处理CSV并直接写出 data_origin.jsonl 与 tokenized_data.jsonl，避免整表入内存"
    )
    parser.add_argument(
        "--text-column-name", help="覆盖 train_config.text_column_name (默认 text)"
    )
    parser.add_argument(
        "--label-column-name", help="覆盖 train_config.label_column_name (默认 label)"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--clean-columns",
        action="append",
        help="清洗后CSV需保留的原始列(可多次或用逗号分隔)；也可在配置 preprocess_config.clean_columns 指定",
    )
    # 聚合导出已废弃，改为仅逐文件输出
    parser.add_argument(
        "--clean-output-suffix",
        default="_cleaned.csv",
        help="逐文件清洗输出的文件名后缀，默认 _cleaned.csv",
    )
    parser.add_argument(
        "--clean-csv",
        action="store_true",
        help="仅执行CSV清洗并输出逐文件结果，不进行 JSONL/匹配/分词 等其他流程",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    full_cfg = load_config(config_path)
    if "preprocess_config" not in full_cfg or "train_config" not in full_cfg:
        print("配置文件需包含 'preprocess_config' 与 'train_config' 两个节点。")
        sys.exit(1)

    preprocess_cfg = full_cfg["preprocess_config"]
    train_cfg = full_cfg["train_config"]

    # 覆盖 train_cfg 相关参数
    if args.model:
        train_cfg["model"] = args.model
    if args.max_seq_length:
        train_cfg["max_seq_length"] = args.max_seq_length
    if args.text_column_name:
        train_cfg["text_column_name"] = args.text_column_name
    if args.label_column_name:
        train_cfg["label_column_name"] = args.label_column_name
    # 覆盖 preprocess_cfg 相关参数
    if args.batch_size:
        preprocess_cfg["batch_size"] = args.batch_size
    if args.workers:
        preprocess_cfg["workers"] = args.workers
    # convert files
    convert_files = preprocess_cfg.get("convert_files", [])
    if args.convert_file:
        # 若用户提供则使用提供的列表 (覆盖)
        convert_files = args.convert_file
    preprocess_cfg["convert_files"] = convert_files

    # remove keywords
    remove_keywords = preprocess_cfg.get("remove_keywords", [])
    if args.remove_keyword:
        remove_keywords = list(dict.fromkeys(remove_keywords + args.remove_keyword))
        preprocess_cfg["remove_keywords"] = remove_keywords

    # valid labels 追加
    if args.valid_label:
        added = [v for v in args.valid_label if v]
        preprocess_cfg["valid_labels"] = list(
            dict.fromkeys((preprocess_cfg.get("valid_labels") or []) + added)
        )

    if remove_keywords:
        print(f"数据清洗关键字数: {len(remove_keywords)}")
    else:
        print("未配置数据清洗关键字")

    # 解析清洗CSV的保留列
    clean_columns_cfg = preprocess_cfg.get("clean_columns") or []
    clean_columns: List[str] = list(clean_columns_cfg)
    if args.clean_columns:
        merged: List[str] = []
        for item in args.clean_columns:
            parts = [p.strip() for p in item.split(",") if p and p.strip()]
            merged.extend(parts)
        # 去重但保持顺序
        seen_c: Set[str] = set()
        clean_columns = []
        for c in merged:
            if c not in seen_c:
                seen_c.add(c)
                clean_columns.append(c)
    # 将最终列写回配置对象，便于输出到 metadata
    preprocess_cfg["clean_columns"] = clean_columns

    all_records: List[Dict[str, Any]] = []
    print(f"共需处理 {len(convert_files)} 个文件")

    # ---- 通配符扩展 (支持 *, ?, [], ** ) ----
    expanded_list: List[str] = []
    seen_paths: Set[str] = set()
    base_dir_config_parent = os.path.abspath(os.path.join(os.path.dirname(config_path), ".."))
    cwd = os.getcwd()

    def try_expand(pattern: str) -> List[str]:
        candidates: List[str] = []
        search_patterns = []
        if os.path.isabs(pattern):
            search_patterns.append(pattern)
        else:
            # 相对 config 上级目录
            search_patterns.append(os.path.join(base_dir_config_parent, pattern))
            # 相对当前工作目录
            search_patterns.append(os.path.join(cwd, pattern))
            # 原样 (以防 pattern 自带子路径引用)
            search_patterns.append(pattern)
        matched: List[str] = []
        for sp in search_patterns:
            g = glob.glob(sp, recursive=True)
            if g:
                matched.extend(g)
        # 去重并仅保留文件
        uniq: List[str] = []
        seen_local: Set[str] = set()
        for m in matched:
            if os.path.isfile(m):
                ap = os.path.abspath(m)
                if ap not in seen_local:
                    seen_local.add(ap)
                    uniq.append(ap)
        return uniq

    wildcard_chars = set('*?[]')
    for item in convert_files:
        if any(ch in item for ch in wildcard_chars):
            expanded = try_expand(item)
            if not expanded:
                print(f"通配符未匹配到文件: {item}")
                continue
            print(f"通配符扩展: {item} -> {len(expanded)} 个文件")
            for p in expanded:
                if p not in seen_paths:
                    seen_paths.add(p)
                    expanded_list.append(p)
        else:
            # 非通配符按原逻辑解析路径
            candidate_paths = []
            if os.path.isabs(item):
                candidate_paths.append(item)
            else:
                candidate_paths.append(os.path.join(base_dir_config_parent, item))
                candidate_paths.append(os.path.join(cwd, item))
                candidate_paths.append(item)
            final_path = None
            for cp in candidate_paths:
                if os.path.exists(cp) and os.path.isfile(cp):
                    final_path = os.path.abspath(cp)
                    break
            if final_path is None:
                print(f"警告: 找不到文件 {item}, 已跳过")
                continue
            if final_path not in seen_paths:
                seen_paths.add(final_path)
                expanded_list.append(final_path)

    print(f"通配符展开后文件总数: {len(expanded_list)}")

    # clean-csv 模式：仅逐文件清洗并退出
    if args.clean_csv:
        if not clean_columns:
            print("错误: clean-csv 模式需要指定 --clean-columns 或在配置 preprocess_config.clean_columns 中配置列名")
            sys.exit(2)
        os.makedirs(args.output_dir, exist_ok=True)
        cleaned_csv_files: List[str] = []
        for src in expanded_list:
            base = os.path.splitext(os.path.basename(src))[0]
            out_clean = os.path.join(args.output_dir, f"{base}{args.clean_output_suffix}")
            stats_csv = write_cleaned_csv_one(src, out_clean, clean_columns, remove_keywords)
            cleaned_csv_files.append(out_clean)
            print(f"清洗CSV完成: {os.path.basename(src)} -> {out_clean}  读取 {stats_csv['read']} 行，写出 {stats_csv['written']} 行")
        print("[完成] 数据清洗导出")
        return

    # 逐个文件处理 (使用展开后的列表)
    for i, csv_path in enumerate(expanded_list, 1):
        print(f"\n--- 处理第 {i}/{len(expanded_list)} 个文件 ---")
        print(f"文件路径: {csv_path}")
        try:
            if args.stream:
                # 流式模式下不进行预扫描与计数，后续阶段单次遍历时边扫描边统计
                pass
            else:
                records = process_csv_file(csv_path, remove_keywords)
                print(f"成功处理 {len(records)} 条记录")
                all_records.extend(records)
                print(f"累计记录数: {len(all_records)}")
        except Exception as e:
            print(f"错误: 处理文件 {csv_path} 时出错: {e}")
            continue

    print("\n=== 文件处理完成 ===")
    if not args.stream:
        print(f"总共处理了 {len(all_records)} 条记录")

    # 主分类号匹配
    print("\n--- 开始主分类号匹配 ---")
    cfg_valid = preprocess_cfg.get("valid_labels") or []
    cfg_valid_norm = sorted(
        {normalize_single_ipc(v) for v in cfg_valid if v}, key=lambda x: (-len(x), x)
    )
    print(f"有效前缀数量: {len(cfg_valid_norm)}")

    matched_count = 0
    total_count = 0
    # 输出目录与文件路径
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    raw_output_file_jsonl = os.path.join(output_dir, "data_origin.jsonl")
    tokenized_output_file = os.path.join(output_dir, "tokenized_data.jsonl")
    cleaned_csv_files: List[str] = []

    # 统计分词观测长度与数量
    tokenized_count_observed = 0
    tokenized_max_len_observed = 0

    if args.stream:
        # 以 JSONL 形式写原始与分词结果（生产者-消费者异步写入）
        print("[流式] 输出到 JSONL: data_origin.jsonl / tokenized_data.jsonl")
        # 写线程与队列
        raw_q: Queue = Queue(maxsize=10000)
        tok_q: Queue = Queue(maxsize=10000)
        raw_thr = threading.Thread(target=_writer_worker, args=(raw_output_file_jsonl, raw_q), daemon=True)
        tok_thr = threading.Thread(target=_writer_worker, args=(tokenized_output_file, tok_q), daemon=True)
        raw_thr.start(); tok_thr.start()

        # 为分词准备
        model_name = train_cfg.get("model")
        if not model_name:
            print("错误: 训练配置中缺少 'model' 名称。")
            sys.exit(1)
        max_seq_length = train_cfg.get("max_seq_length", 512)
        batch_size = preprocess_cfg.get("batch_size", 32)
        workers = preprocess_cfg.get("workers", 1)
        tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=True)
        if args.skip_tokenize:
            print(f"[流式] 分词已跳过 (--skip-tokenize)")
        else:
            print(
                f"[流式] 分词配置 -> 线程: {workers}  批大小: {batch_size}  最大长度: {max_seq_length}  模型: {model_name}"
            )

        # 流式聚合一个批
        buf_texts: List[str] = []
        buf_labels: List[int] = []
        buf_ids: List[str] = []

        # 多线程流式分词支持
        use_mt = (not args.skip_tokenize) and workers and workers > 1
        stats_lock = threading.Lock()
        futures = []
        max_inflight = max(2, workers * 2) if use_mt else 0
        ex = ThreadPoolExecutor(max_workers=workers) if use_mt else None

        def _encode_and_enqueue(batch_texts: List[str], batch_labels: List[int], batch_ids: List[str]):
            nonlocal tokenized_count_observed, tokenized_max_len_observed
            enc = tokenizer(
                batch_texts,
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
            )
            local_count = 0
            local_max = 0
            for i, input_ids in enumerate(enc["input_ids"]):
                rec = {
                    "input_ids": input_ids,
                    "attention_mask": enc["attention_mask"][i],
                    "label": batch_labels[i],
                    "original_id": batch_ids[i],
                }
                tok_q.put(json.dumps(rec, ensure_ascii=False) + "\n")
                local_count += 1
                if isinstance(input_ids, list):
                    if len(input_ids) > local_max:
                        local_max = len(input_ids)
            # 线程安全更新统计
            with stats_lock:
                tokenized_count_observed += local_count
                if local_max > tokenized_max_len_observed:
                    tokenized_max_len_observed = local_max

        pbar = None
        try:
            # 流式实际处理：单次遍历，边扫描边统计
            pbar = tqdm(total=None, desc="流式处理", unit="行")
            for csv_path in expanded_list:
                for r in process_csv_file_stream(csv_path, preprocess_cfg.get('remove_keywords')):
                    total_count += 1
                    if pbar is not None:
                        pbar.update(1)
                    ipc_code = normalize_single_ipc(r.get("ipc", ""))
                    matched_prefix = ""
                    for pref in cfg_valid_norm:
                        if ipc_code.startswith(pref):
                            matched_prefix = pref
                            break
                    is_valid = bool(matched_prefix)
                    if is_valid:
                        matched_count += 1
                    r["matched_prefix"] = matched_prefix
                    r["label"] = 1 if is_valid else 0
                    # 入队原始 JSONL 行
                    raw_q.put(json.dumps(r, ensure_ascii=False) + "\n")

                    # 分词缓冲
                    if not args.skip_tokenize:
                        buf_texts.append(r.get("text", ""))
                        buf_labels.append(r["label"])  # type: ignore
                        buf_ids.append(r.get("id", ""))
                        if len(buf_texts) >= batch_size:
                            if use_mt and ex is not None:
                                # 控制飞行中的任务数量，避免占用过多内存
                                while len(futures) >= max_inflight:
                                    # 回收最早一个完成的任务
                                    done = None
                                    for f in futures:
                                        if f.done():
                                            done = f
                                            break
                                    if done is None:
                                        # 若都未完成，阻塞等待第一个完成
                                        futures[0].result()
                                        done = futures[0]
                                    futures.remove(done)
                                futures.append(ex.submit(_encode_and_enqueue, buf_texts, buf_labels, buf_ids))
                            else:
                                _encode_and_enqueue(buf_texts, buf_labels, buf_ids)
                            buf_texts = []
                            buf_labels = []
                            buf_ids = []
            # 处理尾批
            if not args.skip_tokenize and buf_texts:
                if use_mt and ex is not None:
                    futures.append(ex.submit(_encode_and_enqueue, buf_texts, buf_labels, buf_ids))
                else:
                    _encode_and_enqueue(buf_texts, buf_labels, buf_ids)
            # 等待所有并行分词完成
            if use_mt and ex is not None:
                for f in futures:
                    # 捕获异常以便尽早抛出
                    f.result()
        finally:
            # 发送结束信号并等待写线程
            raw_q.put(None)
            tok_q.put(None)
            raw_thr.join(); tok_thr.join()
            if ex is not None:
                ex.shutdown(wait=True, cancel_futures=False)
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass
        # 流式阶段结束后，如配置了保留列，进行逐文件清洗后CSV导出（二次遍历源文件）
        if clean_columns:
            multi = len(expanded_list) > 1
            # 逐文件输出，使用 --clean-output-suffix
            for src in expanded_list:
                base = os.path.splitext(os.path.basename(src))[0]
                out_clean = os.path.join(output_dir, f"{base}{args.clean_output_suffix}")
                stats_csv = write_cleaned_csv_one(src, out_clean, clean_columns, remove_keywords)
                cleaned_csv_files.append(out_clean)
                print(f"清洗CSV完成: {os.path.basename(src)} -> {out_clean}  读取 {stats_csv['read']} 行，写出 {stats_csv['written']} 行")
    else:
        total_count = len(all_records)
        for r in tqdm(all_records, desc="匹配", unit="记录"):
            ipc_code = normalize_single_ipc(r.get("ipc", ""))
            matched_prefix = ""
            for pref in cfg_valid_norm:
                if ipc_code.startswith(pref):
                    matched_prefix = pref
                    break
            is_valid = bool(matched_prefix)
            r["matched_prefix"] = matched_prefix
            r["label"] = 1 if is_valid else 0
            if is_valid:
                matched_count += 1
    print("匹配完成:")
    print(
        f"  匹配成功: {matched_count} ({(matched_count/total_count*100) if total_count else 0:.1f}%)"
    )
    print(
        f"  未匹配: {total_count - matched_count} ({((total_count-matched_count)/total_count*100) if total_count else 0:.1f}%)"
    )

    # 预先定义，便于后续引用
    tokenized_data: List[Dict[str, Any]] = []

    # 分词 (非流式模式)
    if not args.stream:
        if args.skip_tokenize:
            print("跳过分词 (--skip-tokenize)")
        else:
            print("分词目标: 全部记录 (label=1 或 0)")
            target_records = all_records
            tokenized_data = tokenize_and_format(target_records, train_cfg, preprocess_cfg)
            # 非流式直接统计
            tokenized_count_observed = len(tokenized_data)
            try:
                tokenized_max_len_observed = max((len(r.get("input_ids", [])) for r in tokenized_data), default=0)
            except Exception:
                tokenized_max_len_observed = 0

    # 输出 (非流式写法) —— 仅生成 JSONL
    if not args.stream:
        print(f"写出原始数据(JSONL): {raw_output_file_jsonl}")
        with open(raw_output_file_jsonl, "w", encoding="utf-8") as f:
            for rec in tqdm(all_records, desc="写出原始JSONL", unit="条"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if not args.skip_tokenize:
            print(f"写出分词数据: {tokenized_output_file}")
            with open(tokenized_output_file, "w", encoding="utf-8") as f:
                for record in tokenized_data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            print("未生成分词 jsonl (跳过)")

    print("\n=== 处理完成 ===")
    print(f"总记录数: {total_count}")
    print(f"匹配成功记录 (label=1): {matched_count}")
    if args.stream:
        print(f"原始数据输出(JSONL): {raw_output_file_jsonl}")
        if not args.skip_tokenize:
            print(f"分词后数据输出(JSONL): {tokenized_output_file}")
    else:
        if not args.skip_tokenize:
            print(f"分词样本数: {len(tokenized_data)}")
            print(f"分词后数据输出(JSONL): {tokenized_output_file}")
        print(f"原始数据输出(JSONL): {raw_output_file_jsonl}")
        # 非流式模式下，同样支持逐文件清洗后CSV导出（独立二次遍历）
        if clean_columns:
            multi = len(expanded_list) > 1
            # 逐文件输出，使用 --clean-output-suffix
            for src in expanded_list:
                base = os.path.splitext(os.path.basename(src))[0]
                out_clean = os.path.join(output_dir, f"{base}{args.clean_output_suffix}")
                stats_csv = write_cleaned_csv_one(src, out_clean, clean_columns, remove_keywords)
                cleaned_csv_files.append(out_clean)
                print(f"清洗CSV完成: {os.path.basename(src)} -> {out_clean}  读取 {stats_csv['read']} 行，写出 {stats_csv['written']} 行")

    # 写出元数据，供后续 split/pack 复用，跳过扫描
    meta = {
        "version": 1,
        "stream": bool(args.stream),
        "output_dir": output_dir,
        "paths": {
            "origin_json": None,
            "origin_jsonl": raw_output_file_jsonl,
            "tokenized_jsonl": tokenized_output_file if not args.skip_tokenize else None,
            # 不再写入单文件 cleaned_csv 聚合路径
            "cleaned_csv_files": cleaned_csv_files if cleaned_csv_files else None,
        },
        "counts": {
            "total_records": int(total_count),
            "label_1": int(matched_count),
            "label_0": int(total_count - matched_count),
        },
        "valid_labels": list(cfg_valid),
        "tokenize": {
            "enabled": (not args.skip_tokenize),
            "model": train_cfg.get("model"),
            "max_seq_length_cfg": int(train_cfg.get("max_seq_length", 512)),
            "batch_size": preprocess_cfg.get("batch_size", 32),
            "workers": preprocess_cfg.get("workers", 1),
            "observed_max_input_len": int(tokenized_max_len_observed),
            "observed_count": int(tokenized_count_observed),
        },
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"元数据输出: {meta_path}")
    except Exception as e:
        print(f"警告: 写出元数据失败: {e}")


if __name__ == "__main__":
    main()
