echo "Starting training..."

# 基础训练参数
python train.py `
    --model ./model `
    --train_file ./dataset/train.jsonl `
    --validation_file ./dataset/val.jsonl `
    --output_dir ./outputs/bert-classification

# 如果需要自定义参数，可以使用以下示例：
# python train.py `
#     --model ./model `
#     --train_file ./dataset/train.jsonl `
#     --validation_file ./dataset/val.jsonl `
#     --output_dir ./outputs/bert-classification `
#     --per_device_train_batch_size 8 `
#     --per_device_eval_batch_size 8 `
#     --num_train_epochs 5 `
#     --learning_rate 3e-5 `
#     --device cuda `
#     --fp16 `
#     --gradient_checkpointing

echo "Training finished."
