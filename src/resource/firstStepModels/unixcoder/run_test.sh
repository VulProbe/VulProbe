#!/bin/bash

python unixcoder_main.py \
  --output_dir=. \
  --model_name=model_00.bin \
  --model_type=roberta \
  --tokenizer_name=microsoft/unixcoder-base-nine \
  --model_name_or_path=microsoft/unixcoder-base-nine \
  --do_test \
  --test_data_file=VulProbe/src/resource/dataset/c/test.jsonl \
  --epochs 20 \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 00