#!/bin/bash

ratio="0.4"

# folder=test
# python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --mixup --alpha 0.1

folder=baseline
mkdir -p checkpoints/"$folder"
python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --data_ratio "$ratio" | tee checkpoints/"$folder"/log

folder=ft_mixup0_1
mkdir -p checkpoints/"$folder"
python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --mixup --alpha 0.1 --data_ratio "$ratio" | tee checkpoints/"$folder"/log

folder=ft_mixup0_4
mkdir -p checkpoints/"$folder"
python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --mixup --alpha 0.4 --data_ratio "$ratio" | tee checkpoints/"$folder"/log

folder=ft_mixup0_6
mkdir -p checkpoints/"$folder"
python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --mixup --alpha 0.6 --data_ratio "$ratio" | tee checkpoints/"$folder"/log

folder=ft_mixup0_8
mkdir -p checkpoints/"$folder"
python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --mixup --alpha 0.8 --data_ratio "$ratio" | tee checkpoints/"$folder"/log

folder=ft_mixup1_0
mkdir -p checkpoints/"$folder"
python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --mixup --alpha 1.0 --data_ratio "$ratio" | tee checkpoints/"$folder"/log
