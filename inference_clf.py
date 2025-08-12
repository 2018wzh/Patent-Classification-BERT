import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

def predict(texts: List[str], model_path: str, batch_size: int = 8):
    """
    Performs inference on a list of texts.

    Args:
        texts (List[str]): A list of texts to classify.
        model_path (str): Path to the trained model directory.
        batch_size (int): Batch size for inference.

    Returns:
        A list of prediction dictionaries.
    """
    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = model.config.id2label

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
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--text", type=str, help="A single text to classify.")
    parser.add_argument("--input_file", type=str, help="Path to an input file (txt or jsonl with a 'text' key).")
    parser.add_argument("--output_file", type=str, help="Path to save the output (jsonl).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    
    args = parser.parse_args()

    if not args.text and not args.input_file:
        raise ValueError("Either --text or --input_file must be provided.")

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
    predictions = predict(texts_to_predict, args.model_path, args.batch_size)

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
