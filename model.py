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
from mixup import mixup_data

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, vocab_size=None, device='cpu', alpha: float=1.0):
        super().__init__()
        self.bert: BertModel
        self.bert = BertModel.from_pretrained('bert-base-cased')  # type: ignore

        self.fc = nn.Linear(768, vocab_size)

        self.device = device
        self.alpha: float = alpha

    def forward(self, x, y, mixup: bool=False):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x, y = x.to(self.device), y.to(self.device)

        if mixup:
            encoded_layer, _, y_a, y_b, lam = self.bert(x, mixup=True, y=y, output_all_encoded_layers=False)  # do not care pooled output
        else:
            encoded_layer, _, = self.bert(x, output_all_encoded_layers=False)  # do not care pooled output


        # if mixup:
        #     encoded_layer, y_a, y_b, lam = mixup_data(encoded_layer, y, self.alpha, self.device)

        logits = self.fc(encoded_layer)  # (N, T, VOCAB)

        if mixup:
            return logits, y_a, y_b, lam  # type: ignore
        return logits, y
