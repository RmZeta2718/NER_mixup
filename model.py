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
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class Net(nn.Module):
    def __init__(self, vocab_size=None, device='cpu', finetuning=False):
        super().__init__()
        self.bert: BertModel
        self.bert = BertModel.from_pretrained('bert-base-cased')  # type: ignore

        self.fc = nn.Linear(768, vocab_size)

        self.device = device
        self.finetuning = finetuning

    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
