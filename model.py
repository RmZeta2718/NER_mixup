"""
bert-base-cased config: {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "type_vocab_size": 2,
    "vocab_size": 28996
}
"""
from pytorch_pretrained_bert import BertModel

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, vocab_size=None, mixup=False, alpha=0.0, device='cpu'):
        super().__init__()
        self.bert: BertModel
        self.bert = BertModel.from_pretrained(
            'bert-base-cased',
            alpha=alpha,
            device=device,  # for mixup data
        )

        self.fc = nn.Linear(768, vocab_size)

        self.mixup: bool = mixup
        self.device: str = device

    def forward(self, x, y=None):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)


        if self.training:
            encoded_layer, _, y_a, y_b, lam = self.bert(x, mixup=self.mixup, labels=y, output_all_encoded_layers=False)  # do not care pooled output
            logits = self.fc(encoded_layer)  # (N, T, VOCAB)
            return logits, y_a, y_b, lam
        else:  # eval
            encoded_layer, _ = self.bert(x, output_all_encoded_layers=False)
            logits = self.fc(encoded_layer)  # (N, T, VOCAB)
            return logits
        
