import csv
import json
import sys
import os
from typing import Dict, Any, List, Set, Optional
from tqdm import tqdm

DEFAULT_CONFIG_PATH = os.path.join("config", "config.json")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


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
        if keyword in cleaned_text:
            cleaned_text = cleaned_text.replace(keyword, "")

    # 清理多余的空白字符
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text


def unify_record(
    row: Dict[str, str], remove_keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    # 标准化 IPC
    all_ipc_raw = row.get("IPC分类号", "")
    main_ipc_raw = row.get("IPC主分类号", "")
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
    if "UNKNOWN" in label_set and len(label_set) > 1:
        label_set.discard("UNKNOWN")

    # 获取文本内容并进行清洗
    patent_name = row.get("﻿专利名称", "")
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
        "labels": all_ipc_list,
        "label": main_ipc_norm,
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
    # 允许: python preprocess.py 或 python preprocess.py <config_path>
    if len(sys.argv) > 2:
        print("用法: python preprocess.py [config_path]")
        sys.exit(1)
    config_path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    convert_files = cfg.get("convertFiles", [])
    output_file = cfg.get("outputFile")
    if not output_file:
        print("config 缺少 outputFile")
        sys.exit(1)

    # 读取数据清洗配置
    remove_keywords = cfg.get("removeKeywords", [])
    if remove_keywords:
        print(f"数据清洗配置: 将删除 {len(remove_keywords)} 个关键字")
        print(f"关键字列表: {remove_keywords}")
    else:
        print("未配置数据清洗关键字")

    all_records: List[Dict[str, Any]] = []
    print(f"共需处理 {len(convert_files)} 个文件")

    # 逐个文件处理，每个文件完全读入内存后处理
    for i, rel in enumerate(convert_files, 1):
        print(f"\n--- 处理第 {i}/{len(convert_files)} 个文件 ---")

        # 解析文件路径
        csv_path = (
            rel
            if os.path.isabs(rel)
            else os.path.join(os.path.dirname(config_path), "..", rel).replace(
                ".." + os.sep + os.sep, ".." + os.sep
            )
        )
        if not os.path.exists(csv_path):
            csv_path = rel if os.path.isabs(rel) else os.path.join(os.getcwd(), rel)
        if not os.path.exists(csv_path):
            print(f"警告: 找不到文件 {rel}, 已跳过")
            continue

        print(f"文件路径: {csv_path}")

        # 处理单个文件（完全读入内存）
        try:
            records = process_csv_file(csv_path, remove_keywords)
            print(f"成功处理 {len(records)} 条记录")
            all_records.extend(records)
            print(f"累计记录数: {len(all_records)}")
        except Exception as e:
            print(f"错误: 处理文件 {csv_path} 时出错: {e}")
            continue

    print(f"\n=== 文件处理完成 ===")
    print(f"总共处理了 {len(all_records)} 条记录")

    # 批量验证所有记录的标签
    print(f"\n--- 开始验证标签 ---")
    # 仅使用配置的 validLabels 进行前缀模糊匹配
    cfg_valid = cfg.get("validLabels") or []
    cfg_valid_norm = sorted(
        set(normalize_single_ipc(v) for v in cfg_valid if v), key=lambda x: (-len(x), x)
    )
    print(f"有效标签前缀: {cfg_valid_norm}")

    def fuzzy_valid(codes: List[str]) -> bool:
        for c in codes:
            nc = normalize_single_ipc(c)
            for pref in cfg_valid_norm:
                if nc.startswith(pref):
                    return True
        return False

    # 批量验证标签（在内存中处理）
    valid_count = 0
    invalid_count = 0

    for r in tqdm(all_records, desc="验证标签", unit="记录"):
        val = fuzzy_valid(r.get("labels", []))
        r["valid"] = val
        if val:
            valid_count += 1
        else:
            invalid_count += 1

    print(f"标签验证完成:")
    print(f"  有效记录: {valid_count} ({valid_count/len(all_records)*100:.1f}%)")
    print(f"  无效记录: {invalid_count} ({invalid_count/len(all_records)*100:.1f}%)")

    # 写入输出文件
    print(f"\n--- 写入输出文件 ---")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"正在写入输出文件: {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n=== 处理完成 ===")
    print(f"总记录数: {len(all_records)}")
    print(f"有效记录: {valid_count}")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    main()
