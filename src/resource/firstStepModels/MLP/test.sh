#!/bin/bash

python mlp.py \
    --save_path=codebert_mlp_classifier.pth \
    --seed=0

python mlp.py \
    --save_path=codebert_mlp_classifier_11.pth \
    --seed=11

python mlp.py \
    --save_path=codebert_mlp_classifier_22.pth \
    --seed=22

python mlp.py \
    --save_path=codebert_mlp_classifier_33.pth \
    --seed=33

python mlp.py \
    --save_path=codebert_mlp_classifier_44.pth \
    --seed=44