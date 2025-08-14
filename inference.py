import os
import json
import argparse
from typing import List

import torch
from transformers import BertTokenizer, BertForSequenceClassification


def load_model(model_path: str, device, data_parallel: bool = False, freeze_encoder_layers: int = 0):
    tokenizer = BertTokenizer.from_pretrained(model_path, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    if data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if freeze_encoder_layers > 0:
        encoder = getattr(model, 'bert', None)
        if encoder is not None:
            layers = getattr(encoder, 'encoder', None)
            if layers and hasattr(layers, 'layer'):
                for i, layer in enumerate(layers.layer):
                    if i < freeze_encoder_layers:
                        for p in layer.parameters():
                            p.requires_grad = False
    model.eval()
    return tokenizer, model


def predict_texts(texts: List[str], tokenizer, model, max_length: int, batch_size: int = 8):
    device = next(model.parameters()).device
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, max_length=max_length, padding=True, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)
            probs_valid = probs[:, 1].cpu().tolist()
            preds = probs_valid  # 概率即可, 阈值留给调用层
        for t, p in zip(batch_texts, probs_valid):
            results.append({'text': t, 'prob_valid': float(p)})
    return results


def gpu_env_diag():
    print("=== 推理环境诊断 ===")
    print(f"torch: {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  {props.total_memory/1024**3:.1f} GB")
    print("====================")


def main():
    parser = argparse.ArgumentParser(description='Inference for valid classifier')
    parser.add_argument('--model', default='outputs/valid-clf', help='模型目录')
    parser.add_argument('--text', default=None, help='单条文本')
    parser.add_argument('--file', default=None, help='输入文件(.txt: 每行一条 / .jsonl: 含 text 字段 / .json: list[str] 或 list[obj])')
    parser.add_argument('--text-key', default='text')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5, help='判定 valid 的概率阈值')
    parser.add_argument('--output', default=None, help='保存预测结果 (jsonl)')
    parser.add_argument('--data-parallel', action='store_true', help='当 device-map=none 时用 DataParallel')
    parser.add_argument('--freeze-encoder-layers', type=int, default=0)
    parser.add_argument('--overflow-strategy', default='none', choices=['none','duplicate'], help='长文本拆分策略 (推理)')
    parser.add_argument('--gpus', default=None, help='指定可见 GPU (如 "0" 或 "0,1")')
    args = parser.parse_args()

    if not args.text and not args.file:
        raise SystemExit('必须提供 --text 或 --file')

    # 设置 GPU 可见性
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f"设置 CUDA_VISIBLE_DEVICES={args.gpus}")
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    gpu_env_diag()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, model = load_model(args.model, device, args.data_parallel, args.freeze_encoder_layers)

    texts = []
    meta = []
    if args.text:
        texts.append(args.text)
        meta.append({'id': 0})
    if args.file:
        ext = os.path.splitext(args.file)[1].lower()
        with open(args.file, 'r', encoding='utf-8') as f:
            if ext == '.txt':
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        texts.append(line)
                        meta.append({'id': i})
            elif ext == '.jsonl':
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    text = obj.get(args.text_key) if isinstance(obj, dict) else str(obj)
                    if text:
                        texts.append(text)
                        meta.append({'id': obj.get('id', i)} if isinstance(obj, dict) else {'id': i})
            elif ext == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    for i, obj in enumerate(data):
                        if isinstance(obj, dict):
                            text = obj.get(args.text_key)
                            if text:
                                texts.append(text)
                                meta.append({'id': obj.get('id', i)})
                        else:
                            texts.append(str(obj))
                            meta.append({'id': i})
                else:
                    raise ValueError('json 文件必须是 list')
            else:
                raise ValueError('不支持的文件类型')

    if not texts:
        raise SystemExit('没有可预测文本')

    if args.overflow_strategy == 'duplicate':
        # 对每条文本切分多个 segment, 取最大概率或平均
        segment_results = []
        for t in texts:
            tok = tokenizer(t, truncation=True, max_length=args.max_length, return_overflowing_tokens=True, stride=0)
            input_ids_list = tok['input_ids'] if isinstance(tok['input_ids'][0], list) else [tok['input_ids']]
            seg_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids_list]
            seg_preds = predict_texts(seg_texts, tokenizer, model, args.max_length, args.batch_size)
            # 聚合: 取最大概率
            max_prob = max(s['prob_valid'] for s in seg_preds)
            segment_results.append({'text': t, 'prob_valid': max_prob})
        results = segment_results
    else:
        results = predict_texts(texts, tokenizer, model, args.max_length, args.batch_size)

    # 应用阈值
    for r, m in zip(results, meta):
        r.update(m)
        r['pred'] = int(r['prob_valid'] >= args.threshold)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print('[DONE] 结果已保存 ->', args.output)
    else:
        for r in results[:20]:  # 控制台打印前 20 条
            print(json.dumps(r, ensure_ascii=False))
        if len(results) > 20:
            print(f'... 共 {len(results)} 条 (仅显示前20)')


if __name__ == '__main__':
    main()
