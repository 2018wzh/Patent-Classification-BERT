import csv
import json
import sys
import os
import argparse
from typing import Dict, Any, List, Set, Optional
from tqdm import tqdm
from transformers import BertTokenizer

DEFAULT_CONFIG_PATH = os.path.join("config", "config.json")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def tokenize_and_format(records: List[Dict[str, Any]], train_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """批量高速分词 (CPU fast tokenizer)。

    使用 Rust 实现的 fast tokenizer 并分批处理，显著提升吞吐，避免逐条 Python 循环的开销。
    GPU 对 WordPiece/BPE 分词无显著收益；真正加速方式是批量 + fast tokenizer。

    额外可配置项 (train_config):
      - tokenize_batch_size: 每批文本数量 (默认 512)
    """
    model_name = train_config.get("model")
    if not model_name:
        print("错误: 训练配置中缺少 'model' 名称。")
        sys.exit(1)

    max_seq_length = train_config.get("max_seq_length", 512)
    text_column = train_config.get("text_column_name", "text")
    label_column = train_config.get("label_column_name", "label")
    batch_size = int(train_config.get("tokenize_batch_size", 512))

    print("\n--- 开始批量分词 (fast tokenizer, CPU) ---")
    print(f"模型/分词器: {model_name}  最大序列长度: {max_seq_length}  批次: {batch_size}")
    tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=True)

    texts = [rec[text_column] for rec in records]
    labels = [1 if rec.get(label_column) else 0 for rec in records]
    original_ids = [rec.get("id", "") for rec in records]
    total = len(texts)
    out: List[Dict[str, Any]] = []
    for start in tqdm(range(0, total, batch_size), desc="分词", unit="batch"):
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
            out.append({
                "input_ids": input_ids,
                "attention_mask": enc["attention_mask"][i],
                "label": labels[start + i],
                "original_id": original_ids[start + i],
            })
    print(f"分词完成，总样本: {len(out)}")
    return out




def normalize_single_ipc(code: str) -> str:
    if not code:
        return ""
    code = code.strip().upper()
    # 替换全角与奇异字符
    code = code.replace("／", "/").replace("－", "-").replace("–", "-")
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
    parser = argparse.ArgumentParser(description="预处理脚本: CSV -> 规范 JSON + tokenized jsonl")
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='配置文件路径 (含 preprocessConfig/trainConfig)')
    parser.add_argument('--convert-file', action='append', help='追加 / 覆盖要处理的 CSV 文件 (可多次指定)')
    parser.add_argument('--model', help='覆盖 trainConfig.model')
    parser.add_argument('--max-seq-length', type=int, help='覆盖 trainConfig.max_seq_length')
    parser.add_argument('--tokenize-batch-size', type=int, help='覆盖 trainConfig.tokenize_batch_size')
    parser.add_argument('--remove-keyword', action='append', help='追加要删除的噪声关键字 (可多次)')
    parser.add_argument('--valid-label', action='append', help='追加有效标签前缀 (可多次)')
    parser.add_argument('--output-dir', default='dataset', help='输出目录 (默认 dataset)')
    parser.add_argument('--skip-tokenize', action='store_true', help='仅解析与标注 valid, 不进行分词')
    # --valid-only 已弃用: 始终处理全部记录并保留 valid 标记供下游过滤
    parser.add_argument('--text-column-name', help='覆盖 trainConfig.text_column_name (默认 text)')
    parser.add_argument('--label-column-name', help='覆盖 trainConfig.label_column_name (默认 label)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    full_cfg = load_config(config_path)
    if 'preprocessConfig' not in full_cfg or 'trainConfig' not in full_cfg:
        print("配置文件需包含 'preprocessConfig' 与 'trainConfig' 两个节点。")
        sys.exit(1)

    preprocess_cfg = full_cfg['preprocessConfig']
    train_cfg = full_cfg['trainConfig']

    # 覆盖 train_cfg 相关参数
    if args.model:
        train_cfg['model'] = args.model
    if args.max_seq_length:
        train_cfg['max_seq_length'] = args.max_seq_length
    if args.tokenize_batch_size:
        train_cfg['tokenize_batch_size'] = args.tokenize_batch_size
    if args.text_column_name:
        train_cfg['text_column_name'] = args.text_column_name
    if args.label_column_name:
        train_cfg['label_column_name'] = args.label_column_name

    # convert files
    convert_files = preprocess_cfg.get("convertFiles", [])
    if args.convert_file:
        # 若用户提供则使用提供的列表 (覆盖)
        convert_files = args.convert_file
    preprocess_cfg['convertFiles'] = convert_files

    # remove keywords
    remove_keywords = preprocess_cfg.get("removeKeywords", [])
    if args.remove_keyword:
        remove_keywords = list(dict.fromkeys(remove_keywords + args.remove_keyword))
        preprocess_cfg['removeKeywords'] = remove_keywords

    # valid labels 追加
    if args.valid_label:
        added = [v for v in args.valid_label if v]
        preprocess_cfg['validLabels'] = list(dict.fromkeys((preprocess_cfg.get('validLabels') or []) + added))

    if remove_keywords:
        print(f"数据清洗关键字数: {len(remove_keywords)}")
    else:
        print("未配置数据清洗关键字")

    all_records: List[Dict[str, Any]] = []
    print(f"共需处理 {len(convert_files)} 个文件")

    # 逐个文件处理
    for i, rel in enumerate(convert_files, 1):
        print(f"\n--- 处理第 {i}/{len(convert_files)} 个文件 ---")
        csv_path = (
            rel if os.path.isabs(rel) else os.path.join(os.path.dirname(config_path), '..', rel)
        )
        if not os.path.exists(csv_path):
            csv_path = rel if os.path.isabs(rel) else os.path.join(os.getcwd(), rel)
        if not os.path.exists(csv_path):
            print(f"警告: 找不到文件 {rel}, 已跳过")
            continue
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
    cfg_valid = preprocess_cfg.get("validLabels") or []
    cfg_valid_norm = sorted({normalize_single_ipc(v) for v in cfg_valid if v}, key=lambda x: (-len(x), x))
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
    print(f"  匹配成功: {matched_count} ({(matched_count/total_count*100) if total_count else 0:.1f}%)")
    print(f"  未匹配: {total_count - matched_count} ({((total_count-matched_count)/total_count*100) if total_count else 0:.1f}%)")

    # 分词
    if args.skip_tokenize:
        tokenized_data: List[Dict[str, Any]] = []
        print("跳过分词 (--skip-tokenize)")
    else:
        print("分词目标: 全部记录 (label=1 或 0)")
        target_records = all_records
        tokenized_data = tokenize_and_format(target_records, train_cfg)

    # 输出
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    raw_output_file = os.path.join(output_dir, "data_origin.json")
    tokenized_output_file = os.path.join(output_dir, "tokenized_data.jsonl")

    # 写出原始: 若指定 --only-valid 则只输出 valid 记录
    origin_to_dump = all_records
    print(f"写出原始数据 (全部记录): {raw_output_file}")
    with open(raw_output_file, 'w', encoding='utf-8') as f:
        json.dump(origin_to_dump, f, ensure_ascii=False, indent=2)

    if not args.skip_tokenize:
        print(f"写出分词数据: {tokenized_output_file}")
        with open(tokenized_output_file, 'w', encoding='utf-8') as f:
            for record in tokenized_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
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
