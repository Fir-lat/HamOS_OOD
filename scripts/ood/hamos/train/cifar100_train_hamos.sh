#!/bin/bash
# sh scripts/ood/hamos/train/cifar100_train_hamos.sh

export CUDA_VISIBLE_DEVICES=0

select=1
loss_weight=0.1
bandwidth=2.0
leapfrog=3
step_size=0.1
margin=0.2
num_neighbor=4

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
            configs/networks/hamos_net.yml \
            configs/pipelines/train/train_hamos.yml \
            configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name hamos \
    --network.backbone.name resnet34_32x32 \
    --network.backbone.checkpoint ./results/cifar100/best.ckpt \
    --dataset.train.batch_size 128 \
    --trainer.trainer_args.temp 0.1 \
    --trainer.trainer_args.K 200 \
    --trainer.trainer_args.select ${select} \
    --trainer.trainer_args.loss_weight ${loss_weight} \
    --trainer.trainer_args.bandwidth ${bandwidth} \
    --trainer.trainer_args.leapfrog ${leapfrog} \
    --trainer.trainer_args.step_size ${step_size} \
    --trainer.trainer_args.margin ${margin} \
    --trainer.trainer_args.num_neighbor ${num_neighbor} \
    --trainer.trainer_args.start_epoch 0 \
    --optimizer.num_epochs 20 \
    --optimizer.lr 0.01 \
    --seed 42


done

