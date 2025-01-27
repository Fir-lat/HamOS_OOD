#!/bin/bash
# sh scripts/basics/cifar100/train_base_backbone_cifar100.sh

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    --seed 42
