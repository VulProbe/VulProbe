#!/bin/bash

python mlp.py \
    --do_train \
    --save_path=codebert_mlp_classifier_123456.pth \
    --seed=123456

# python mlp.py \
#     --do_train \
#     --save_path=codebert_mlp_classifier_11.pth \
#     --seed=11

# python mlp.py \
#     --do_train \
#     --save_path=codebert_mlp_classifier_22.pth \
#     --seed=22

# python mlp.py \
#     --do_train \
#     --save_path=codebert_mlp_classifier_33.pth \
#     --seed=33

# python mlp.py \
#     --do_train \
#     --save_path=codebert_mlp_classifier_44.pth \
#     --seed=44