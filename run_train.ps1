echo "Starting training..."
python train.py `
    --model ./model `
    --train_file ./dataset/train.jsonl `
    --validation_file ./dataset/val.jsonl `
    --label2id_file ./dataset/label2id.json

echo "Training finished."
