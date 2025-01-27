#!/bin/bash
# sh scripts/basics/cifar10/train_base_backbone_cifar10.sh

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    --seed 42
