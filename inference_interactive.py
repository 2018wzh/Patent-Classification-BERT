import os
import json
import argparse
import time
from typing import List

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 交互式推理脚本
# 特性:
#  1. 启动后加载模型与分词器，仅初始化一次
#  2. 支持行级快速输入: 直接输入一行文本回车即可得到预测
#  3. 支持多行模式: 输入 /begin 后连续输入多行，/end 结束整合为一个样本
#  4. 支持基本命令:
#       /q 或 /quit 或 /exit  退出
#       /help                  查看帮助
#       /threshold <v>         动态修改判定阈值 (0-1)
#       /maxlen <n>            修改最大截断长度
#       /file <path>           读取文件批量推理 (txt 每行一条, jsonl 含 text 字段, json list[str|obj])
#       /showcfg               显示当前配置
#       /begin /end            多行输入块
#  5. 输出: 概率、分类(0/1)、耗时(ms) 以及可选保存最近一次结果到历史
#  6. 不新增依赖, 仅使用标准库 + torch + transformers

ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"
ANSI_YELLOW = "\033[93m"


def gpu_env_diag():
    print("=== 推理环境诊断 ===")
    print(f"torch: {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  {props.total_memory/1024**3:.1f} GB")
    print("====================")


def load_model(model_path: str, device, data_parallel: bool = False):
    tokenizer = BertTokenizer.from_pretrained(model_path, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    if data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    return tokenizer, model


def predict(texts: List[str], tokenizer, model, max_length: int) -> List[float]:
    device = next(model.parameters()).device
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().tolist()
    return probs


def read_file_batch(path: str, text_key: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    texts: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        if ext == '.txt':
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        elif ext == '.jsonl':
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    t = obj.get(text_key)
                else:
                    t = str(obj)
                if t:
                    texts.append(t)
        elif ext == '.json':
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError('json 顶层需为 list')
            for obj in data:
                if isinstance(obj, dict):
                    t = obj.get(text_key)
                else:
                    t = str(obj)
                if t:
                    texts.append(t)
        else:
            raise ValueError('不支持的文件类型 (支持 .txt/.jsonl/.json)')
    return texts


def print_help():
    print("\n可用命令:")
    print("  /q | /quit | /exit           退出")
    print("  /help                        显示帮助")
    print("  /threshold <v>               修改阈值 (0-1)")
    print("  /maxlen <n>                  修改最大序列长度")
    print("  /file <path>                 批量推理文件内容")
    print("  /showcfg                     显示当前配置")
    print("  /begin                       进入多行输入模式 (以 /end 结束)")
    print("  /end                         结束多行输入模式并执行预测")
    print("  直接输入任意文本行回车进行预测")
    print()


def interactive_loop(tokenizer, model, max_length: int, threshold: float, text_key: str):
    current_max_length = max_length
    current_threshold = threshold
    multiline_buffer: List[str] = []
    multiline_mode = False
    print_help()
    print(f"进入交互模式: max_length={current_max_length} threshold={current_threshold}")
    while True:
        try:
            inp = input(">> ").rstrip('\n')
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not inp and not multiline_mode:
            continue
        if inp.startswith('/') and not multiline_mode:
            parts = inp.split()
            cmd = parts[0].lower()
            if cmd in {'/q', '/quit', '/exit'}:
                print('再见。')
                break
            elif cmd == '/help':
                print_help()
                continue
            elif cmd == '/threshold' and len(parts) == 2:
                try:
                    v = float(parts[1])
                    if not (0 <= v <= 1):
                        raise ValueError
                    current_threshold = v
                    print(f"阈值已更新: {current_threshold}")
                except ValueError:
                    print('无效阈值，需在 0-1 之间')
                continue
            elif cmd == '/maxlen' and len(parts) == 2:
                try:
                    n = int(parts[1])
                    if n <= 0:
                        raise ValueError
                    current_max_length = n
                    print(f"max_length 已更新: {current_max_length}")
                except ValueError:
                    print('无效 maxlen (需正整数)')
                continue
            elif cmd == '/file' and len(parts) >= 2:
                path = ' '.join(parts[1:])
                if not os.path.exists(path):
                    print('文件不存在')
                    continue
                texts = read_file_batch(path, text_key)
                if not texts:
                    print('文件中无可用文本')
                    continue
                print(f"读取 {len(texts)} 条文本, 正在推理...")
                t0 = time.time()
                probs = predict(texts, tokenizer, model, current_max_length)
                dt = (time.time() - t0) * 1000
                for i, (t, p) in enumerate(zip(texts[:50], probs[:50])):
                    pred = 1 if p >= current_threshold else 0
                    color = ANSI_GREEN if pred == 1 else ANSI_RED
                    print(f"[{i}] {color}{p:.4f}{ANSI_RESET} pred={pred} text={t[:120].replace('\n',' ')}{'...' if len(t)>120 else ''}")
                if len(texts) > 50:
                    print(f"... 共 {len(texts)} 条 (仅显示前50)")
                print(f"批量耗时: {dt:.1f} ms  平均: {dt/len(texts):.2f} ms/样本")
                continue
            elif cmd == '/showcfg':
                print(f"当前: max_length={current_max_length} threshold={current_threshold} device={next(model.parameters()).device}")
                continue
            elif cmd == '/begin':
                multiline_mode = True
                multiline_buffer.clear()
                print('进入多行模式，输入 /end 结束。')
                continue
            elif cmd == '/end':
                print('当前不在多行模式。')
                continue
            else:
                print('未知命令, 输入 /help 查看帮助。')
                continue
        else:
            if multiline_mode:
                if inp.strip().lower() == '/end':
                    text_block = '\n'.join(multiline_buffer).strip()
                    multiline_mode = False
                    if not text_block:
                        print('空多行输入，忽略。')
                        continue
                    t0 = time.time()
                    probs = predict([text_block], tokenizer, model, current_max_length)
                    dt = (time.time() - t0) * 1000
                    p = probs[0]
                    pred = 1 if p >= current_threshold else 0
                    color = ANSI_GREEN if pred == 1 else ANSI_RED
                    print(f"多行样本 -> prob={color}{p:.4f}{ANSI_RESET} pred={pred}  len={len(text_block)}  耗时={dt:.1f} ms")
                    multiline_buffer.clear()
                else:
                    multiline_buffer.append(inp)
                continue
            # 单行文本预测
            text = inp.strip()
            if not text:
                continue
            t0 = time.time()
            probs = predict([text], tokenizer, model, current_max_length)
            dt = (time.time() - t0) * 1000
            p = probs[0]
            pred = 1 if p >= current_threshold else 0
            color = ANSI_GREEN if pred == 1 else ANSI_RED
            print(f"prob={color}{p:.4f}{ANSI_RESET} pred={pred} len={len(text)} 耗时={dt:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description='交互式推理 (单条/批量/多行)')
    parser.add_argument('--model', default='outputs/valid-clf', help='模型目录')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--gpus', default=None, help='指定可见 GPU (如 "0" 或 "0,1")')
    parser.add_argument('--text-key', default='text', help='文件批量模式下的文本字段名')
    parser.add_argument('--data-parallel', action='store_true', help='多 GPU 使用 DataParallel')
    args = parser.parse_args()

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f"设置 CUDA_VISIBLE_DEVICES={args.gpus}")
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    gpu_env_diag()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, model = load_model(args.model, device, args.data_parallel)

    interactive_loop(tokenizer, model, args.max_length, args.threshold, args.text_key)


if __name__ == '__main__':
    main()
