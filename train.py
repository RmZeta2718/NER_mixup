from utils import progress_bar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from data_load import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
import os
import numpy as np
import argparse
from objprint import op
import logging

from mixup import mixup_criterion


def main():
    op.config(color=True, line_number=True, arg_name=True)
    op.install()
    logging.basicConfig(level=logging.INFO)
    data_dir = 'data/CoNLL2003'
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints")
    parser.add_argument("--trainset", type=str, default=f"{data_dir}/train.txt")
    parser.add_argument("--validset", type=str, default=f"{data_dir}/valid.txt")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--mixup", dest="mixup", action="store_true")
    parser.add_argument("--original", dest="original", action="store_true")
    parser.add_argument("--data_ratio", type=float, default=1.0)
    hp = parser.parse_args()

    if hp.seed != 0:
        torch.manual_seed(hp.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    model = Net(len(VOCAB), device).cuda(device)  # type: ignore
    # model = nn.DataParallel(model)

    train_dataset = NerDataset(hp.trainset, hp.data_ratio)
    eval_dataset = NerDataset(hp.validset)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, hp.n_epochs+1):
        # eval(model, eval_iter, "test")
        # exit()
        train(model, train_iter, optimizer, criterion, hp.mixup, hp.original)

        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(hp.logdir):
            os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, str(epoch))
        precision, recall, f1 = eval(model, eval_iter, fname)

        torch.save(model.state_dict(), f"{fname}.pt")
        print(f"weights were saved to {fname}.pt")


def train(model: nn.Module, iterator: data.DataLoader, optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss, mixup: bool, original: bool):
    model.train()

    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y  # for monitoring

        logits: torch.Tensor
        y: torch.Tensor
        if mixup:
            optimizer.zero_grad()
            logits, y_a, y_b, lam = model(x, y, mixup=True)
            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y_a = y_a.view(-1)  # (N*T, 1)
            y_b = y_b.view(-1)  # (N*T, 1)
            loss: torch.Tensor = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()

            optimizer.step()

        # run original data
        if not mixup or original:  # no mixup: directly run, `original` flag on: run original on mixup
            optimizer.zero_grad()
            logits, y = model(x, y)  # logits: (N, T, VOCAB), y: (N, T)

            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y = y.view(-1)  # (N*T, 1)

            loss: torch.Tensor = criterion(logits, y)
            loss.backward()

            optimizer.step()

        if i == 0:
            print("=====sanity check======")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")

        if i % 10 == 0 or i + 1 == len(iterator):  # monitoring
            progress_bar(i, len(iterator), f"step: {i} | loss: {loss.item()}")


def eval(model: nn.Module, iterator: data.DataLoader, f):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, (words, x, is_heads, tags, y, _) in enumerate(iterator):

            logits, _ = model(x, y)  # logits: (N, T, VOCAB)
            y_hat = logits.argmax(-1)  # y_hat: (N, T)
            # op(logits.shape, y_hat.shape)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # gets results and save
    with open("temp", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    # calc metric
    y_true = np.array([tag2idx[line.split()[1]] for line in open(
        "temp", 'r').read().splitlines() if len(line) > 0])
    y_pred = np.array([tag2idx[line.split()[2]] for line in open(
        "temp", 'r').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred > 1])
    num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(int).sum()
    num_gold = len(y_true[y_true > 1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    final = f + ".P%.2f_R%.2f_F%.2f" % (precision, recall, f1)
    with open(final, 'w') as fout:
        result = open("temp", "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision:.5f}\n")
        fout.write(f"recall={recall:.5f}\n")
        fout.write(f"f1={f1:.5f}\n")

    os.remove("temp")

    print(f"precision={precision:.5f}\n")
    print(f"recall={recall:.5f}\n")
    print(f"f1={f1:.5f}\n")
    return precision, recall, f1


if __name__ == "__main__":
    main()