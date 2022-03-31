#!/bin/bash

python train.py --logdir checkpoints/finetuning --batch_size 32 --seed 123456 | tee checkpoints/finetuning/log