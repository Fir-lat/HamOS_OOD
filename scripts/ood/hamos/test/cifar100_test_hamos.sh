#!/bin/bash
# sh scripts/ood/hamos/test/cifar100_test_hamos.sh

export CUDA_VISIBLE_DEVICES=0

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_hamos_net_hamos_e20_lr0.01_default \
   --postprocessor hamos \
   --save-score --save-csv

