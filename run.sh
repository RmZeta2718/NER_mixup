#!/bin/bash

# python train.py --logdir checkpoints/finetuning --batch_size 16 --seed 2718281828 | tee checkpoints/finetuning/log

folder=ft_mixup0_1
mkdir -p checkpoints/"$folder"
python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --mixup --alpha 0.1 | tee checkpoints/"$folder"/log

# folder=ft_mixup0_4
# mkdir -p checkpoints/"$folder"
# python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --mixup --alpha 0.4 | tee checkpoints/"$folder"/log

# folder=ft_mixup0_6
# mkdir -p checkpoints/"$folder"
# python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --mixup --alpha 0.6 | tee checkpoints/"$folder"/log

# folder=ft_mixup0_8
# mkdir -p checkpoints/"$folder"
# python train.py --logdir checkpoints/"$folder" --batch_size 16 --seed 2718281828 --mixup --alpha 0.8 | tee checkpoints/"$folder"/log
