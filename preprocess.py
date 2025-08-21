import csv
import json
import sys
import os
import argparse
import glob
import math
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
    batch_size = int(preprocess_config.get("batch_size", 512))
    workers = int(preprocess_config.get("workers", 1) or 1)
    if workers < 1:
        workers = 1
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

    grant_no = row.get("授权公告号", "").strip()
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
        all_rows = list(reader)

    print(f"文件已读入内存，共 {len(all_rows)} 行数据")

    # 第二步：在内存中处理所有行
    data: List[Dict[str, Any]] = []
    for row in tqdm(all_rows, desc=f"处理 {os.path.basename(path)}", unit="行"):
        processed_record = unify_record(row, remove_keywords)
        data.append(processed_record)

    return data


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
        "--tokenize-batch-size", type=int, help="覆盖 train_config.tokenize_batch_size"
    )
    parser.add_argument(
        "--tokenize-workers", type=int, help="分词线程数 (覆盖 train_config.tokenize_workers, 默认1)"
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
    # --valid-only 已弃用: 始终处理全部记录并保留 valid 标记供下游过滤
    parser.add_argument(
        "--text-column-name", help="覆盖 train_config.text_column_name (默认 text)"
    )
    parser.add_argument(
        "--label-column-name", help="覆盖 train_config.label_column_name (默认 label)"
    )
    parser.add_argument("--verbose", action="store_true")
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
    if args.tokenize_batch_size:
        train_cfg["tokenize_batch_size"] = args.tokenize_batch_size
    if args.tokenize_workers:
        train_cfg["tokenize_workers"] = args.tokenize_workers
    if args.text_column_name:
        train_cfg["text_column_name"] = args.text_column_name
    if args.label_column_name:
        train_cfg["label_column_name"] = args.label_column_name

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

    # 逐个文件处理 (使用展开后的列表)
    for i, csv_path in enumerate(expanded_list, 1):
        print(f"\n--- 处理第 {i}/{len(expanded_list)} 个文件 ---")
        print(f"文件路径: {csv_path}")
        try:
            records = process_csv_file(csv_path, remove_keywords)
            print(f"成功处理 {len(records)} 条记录")
            all_records.extend(records)
            print(f"累计记录数: {len(all_records)}")
        except Exception as e:
            print(f"错误: 处理文件 {csv_path} 时出错: {e}")
            continue

    print("\n=== 文件处理完成 ===")
    print(f"总共处理了 {len(all_records)} 条记录")

    # 主分类号匹配
    print("\n--- 开始主分类号匹配 ---")
    cfg_valid = preprocess_cfg.get("valid_labels") or []
    cfg_valid_norm = sorted(
        {normalize_single_ipc(v) for v in cfg_valid if v}, key=lambda x: (-len(x), x)
    )
    print(f"有效前缀数量: {len(cfg_valid_norm)}")

    matched_count = 0
    total_count = len(all_records)
    for r in tqdm(all_records, desc="匹配", unit="记录"):
        ipc_code = normalize_single_ipc(r.get("ipc", ""))
        matched_prefix = ""
        for pref in cfg_valid_norm:
            if ipc_code.startswith(pref):
                matched_prefix = pref
                break
        is_valid = bool(matched_prefix)
        r["matched_prefix"] = matched_prefix  # 保存匹配到的前缀 (若无匹配为空串)
        # 不再输出 valid 字段; 仅使用 label=1/0 表示有效性
        r["label"] = 1 if is_valid else 0  # 数值标签 (二分类: 1=valid,0=invalid)
        if is_valid:
            matched_count += 1
    print("匹配完成:")
    print(
        f"  匹配成功: {matched_count} ({(matched_count/total_count*100) if total_count else 0:.1f}%)"
    )
    print(
        f"  未匹配: {total_count - matched_count} ({((total_count-matched_count)/total_count*100) if total_count else 0:.1f}%)"
    )

    # 分词
    if args.skip_tokenize:
        tokenized_data: List[Dict[str, Any]] = []
        print("跳过分词 (--skip-tokenize)")
    else:
        print("分词目标: 全部记录 (label=1 或 0)")
        target_records = all_records
        batch_size = preprocess_cfg.get("batchSize", 512)
        tokenized_data = tokenize_and_format(target_records, train_cfg, preprocess_cfg)

    # 输出
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    raw_output_file = os.path.join(output_dir, "data_origin.json")
    tokenized_output_file = os.path.join(output_dir, "tokenized_data.jsonl")

    # 写出原始: 若指定 --only-valid 则只输出 valid 记录
    origin_to_dump = all_records
    print(f"写出原始数据 (全部记录): {raw_output_file}")
    with open(raw_output_file, "w", encoding="utf-8") as f:
        json.dump(origin_to_dump, f, ensure_ascii=False, indent=2)

    if not args.skip_tokenize:
        print(f"写出分词数据: {tokenized_output_file}")
        with open(tokenized_output_file, "w", encoding="utf-8") as f:
            for record in tokenized_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        print("未生成分词 jsonl (跳过)")

    print("\n=== 处理完成 ===")
    print(f"总记录数: {total_count}")
    # 兼容旧输出字段名称, 使用 matched_count
    print(f"匹配成功记录 (label=1): {matched_count}")
    if not args.skip_tokenize:
        print(f"分词样本数: {len(tokenized_data)}")
        print(f"分词后数据输出: {tokenized_output_file}")
    print(f"原始数据输出: {raw_output_file}")


if __name__ == "__main__":
    main()
