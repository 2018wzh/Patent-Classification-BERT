#!/bin/bash

echo "Preprocessing..."
python preprocess.py $1
echo "Preprocessing finished."

echo "Splitting dataset..."
python split_dataset.py $1
echo "Dataset split finished."

echo "Packaging dataset..."
python pack_dataset.py $1
echo "Packaging finished."

echo "Starting training..."
python train.py $1
echo "Training finished."

