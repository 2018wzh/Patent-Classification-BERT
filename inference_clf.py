import argparse
import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List


def gpu_env_diag():
    print("=== 推理环境诊断 ===")
    print(f"torch: {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  {props.total_memory/1024**3:.1f} GB")
    print("====================")

def predict(texts: List[str], model: str, batch_size: int = 8):
    """
    Performs inference on a list of texts.

    Args:
        texts (List[str]): A list of texts to classify.
        model (str): Path to the trained model directory.
        batch_size (int): Batch size for inference.

    Returns:
        A list of prediction dictionaries.
    """
    # 1. Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model)
    model = BertForSequenceClassification.from_pretrained(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = model.config.id2label if model.config.id2label else {0: "false", 1: "true"}

    results = []
    # 2. Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            
        # 3. Get predictions and probabilities
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_ids = torch.argmax(logits, dim=-1)

        # Move to CPU for post-processing
        predicted_class_ids = predicted_class_ids.cpu().numpy()
        probabilities = probabilities.cpu().numpy()

        for text, pred_id, probs in zip(batch_texts, predicted_class_ids, probabilities):
            pred_label = id2label[pred_id]
            prob_dist = {id2label[j]: round(float(p), 4) for j, p in enumerate(probs)}
            
            results.append({
                "text": text,
                "predicted_label": pred_label,
                "predicted_id": int(pred_id),
                "probabilities": prob_dist
            })
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Inference script for sequence classification.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--text", type=str, help="A single text to classify.")
    parser.add_argument("--input_file", type=str, help="Path to an input file (txt or jsonl with a 'text' key).")
    parser.add_argument("--output_file", type=str, help="Path to save the output (jsonl).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument('--gpus', default=None, help='指定可见 GPU (如 "0" 或 "0,1")')
    
    args = parser.parse_args()

    if not args.text and not args.input_file:
        raise ValueError("Either --text or --input_file must be provided.")

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f"设置 CUDA_VISIBLE_DEVICES={args.gpus}")
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    gpu_env_diag()

    # Collect texts to predict
    texts_to_predict = []
    if args.text:
        texts_to_predict.append(args.text)
    if args.input_file:
        if args.input_file.endswith(".txt"):
            with open(args.input_file, 'r', encoding='utf-8') as f:
                texts_to_predict.extend([line.strip() for line in f if line.strip()])
        elif args.input_file.endswith(".jsonl"):
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        texts_to_predict.append(json.loads(line)['text'])
        else:
            raise ValueError("Unsupported file format. Use .txt or .jsonl.")

    # Get predictions
    predictions = predict(texts_to_predict, args.model, args.batch_size)

    # Output results
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        print(f"Predictions saved to {args.output_file}")
    else:
        for pred in predictions:
            print(json.dumps(pred, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
