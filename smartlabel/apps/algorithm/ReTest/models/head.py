import torch
import torch.nn as nn
from typing import Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import init_weights


# non-linear projection head
class NoneLinearProjectionHead(nn.Module):
    def __init__(self, dim_in: int=2048, dim_out: int=128, dim_hidden: int=2048):
        super().__init__() # type: ignore
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.relu1 = nn.ReLU(True)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden)
        self.relu2 = nn.ReLU(True)
        self.linear3 = nn.Linear(dim_hidden, dim_out)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x).unsqueeze(-1).unsqueeze(-1)
        x = self.bn1(x).squeeze(-1).squeeze(-1)
        x = self.relu1(x)
        x = self.linear2(x).unsqueeze(-1).unsqueeze(-1)
        x = self.bn2(x).squeeze(-1).squeeze(-1)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    
class MLPHead(nn.Module):
    def __init__(self, in_channels: int, mlp_hidden: float, projection_size: int, init_method: str='He', with_BN: bool=True, projection_dim: Union[int, None]= None):
        super().__init__() # type: ignore
        if projection_dim is None:
            mlp_hidden_size = round(mlp_hidden * in_channels)
        else:
            mlp_hidden_size = projection_dim
        if with_BN:
            self.mlp_head = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
            )
        init_weights(self.mlp_head, init_method)

    def forward(self, x: torch.Tensor, projection_ret: bool= False):
        if not projection_ret:
            return self.mlp_head(x)
        else:
            projection = self.mlp_head[:-1](x)
            logits = self.mlp_head[-1](projection)
            return projection, logits