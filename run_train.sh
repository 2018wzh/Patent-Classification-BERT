#!/bin/bash

echo "Preprocessing..."
python preprocess.py
echo "Preprocessing finished."

echo "Splitting dataset..."
python split_dataset.py
echo "Dataset split finished."

echo "Starting training..."
python train.py
echo "Training finished."

