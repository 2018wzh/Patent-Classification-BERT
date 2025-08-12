echo "Starting training..."
python train.py `
    --model ./model `
    --train_file ./dataset/train.jsonl `
    --validation_file ./dataset/val.jsonl `
    --output_dir ./outputs/bert-classification

echo "Training finished."
