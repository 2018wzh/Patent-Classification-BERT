#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <num_gpus>"
  exit 1
fi

echo "$1 GPUs configured"

echo "Preprocessing..."
python preprocess.py $2
echo "Preprocessing finished."

echo "Splitting dataset..."
python split_dataset.py $2
echo "Dataset split finished."

echo "Packaging dataset..."
python pack_dataset.py $2
echo "Packaging finished."

echo "Starting training on $1 gpus.."
torchrun --nproc-per-node $1 train.py $2
echo "Training finished."

