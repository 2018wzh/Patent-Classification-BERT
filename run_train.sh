#!/bin/bash

echo "Starting training..."

python train.py \
     --model ./model \
     --train_file ./dataset/train.jsonl \
     --validation_file ./dataset/val.jsonl \
     --output_dir ./outputs/bert-classification \
     --per_device_train_batch_size 8 \
     --per_device_eval_batch_size 8 \
     --num_train_epochs 5 \
     --learning_rate 3e-5 \
     --device cuda \
     --fp16 \
     --gpus 0 \
     --gradient_checkpointing

echo "Training finished."
