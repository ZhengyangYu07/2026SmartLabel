from torch import nn
import torch
from typing import Union, Any
import numpy as np


class NewsNet(nn.Module):
    def __init__(self, weights_matrix: Union[torch.Tensor, np.ndarray[Any, Any]], context_size: int=1000, hidden_size: int=300, num_classes: int=7):
        super(NewsNet, self).__init__() # type: ignore
        n_embed, d_embed = weights_matrix.shape
        self.embedding = nn.Embedding(n_embed, d_embed)
        self.embedding.weight.data.copy_(torch.Tensor(weights_matrix))
        self.avgpool=nn.AdaptiveAvgPool1d(16*hidden_size)
        self.fc1 = nn.Linear(16*hidden_size, 4*hidden_size)
        self.bn1=nn.BatchNorm1d(4*hidden_size)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.bn2=nn.BatchNorm1d(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(x) # input (128, 1000)
        embed = embed.detach()    # embed (128, 1000, 300)
        out = embed.view((1, embed.size()[0], -1)) # (1, 128, 300 000)
        out = self.avgpool(out)
        out = out.squeeze(0)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.classifier(out)
        return out