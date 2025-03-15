#!/bin/bash

python bert.py \
  --model_name=model_00.bin \
  --output_dir=. \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_test \
  --train_data_file=VulProbe/src/resource/dataset/c/train.jsonl \
  --eval_data_file=VulProbe/src/resource/dataset/c/valid.jsonl \
  --test_data_file=VulProbe/src/resource/dataset/c/test.jsonl \
  --epochs 20 \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 00
