#!/bin/bash

meta=https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/metadata
train=https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/train.txt
valid=https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/valid.txt
test=https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/test.txt

data_dir="data/CoNLL2003"
mkdir -p $data_dir
wget --show-progress $meta && mv metadata $data_dir
wget --show-progress $train && mv train.txt $data_dir
wget --show-progress $valid && mv valid.txt $data_dir
wget --show-progress $test && mv test.txt $data_dir