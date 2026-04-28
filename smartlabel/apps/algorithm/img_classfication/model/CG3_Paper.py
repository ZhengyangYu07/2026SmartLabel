import torch
import torch.nn as nn
import torch.nn.functional as F


def graph_matmul(A, X):
    return torch.sparse.mm(A, X.to(A.device)) if A.is_sparse else torch.matmul(A, X.to(A.device))


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=F.relu, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, X, A):
        X = graph_matmul(A, X)
        X = self.fc(X)
        return self.activation(X) if self.activation else X

#多层GCN堆栈（堆叠多个GCN层构建深层网络）
class GCNStack(nn.Module):
    def __init__(self, dims, activation=F.relu):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            act = activation if i < len(dims) - 2 else None
            self.layers.append(GCNLayer(dims[i], dims[i + 1], activation=act))

    def forward(self, X, A):
        for layer in self.layers:
            X = layer(X, A)
        return X


class CG3_Paper(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_classes=4, num_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.local_encoder = GCNStack(dims)
        self.global_encoder = GCNStack(dims)
        self.classifier = nn.Linear(out_dim, num_classes)

    #双路径GCN分别学习局部信息和全局信息
    def forward(self, X, A, A_high=None):
        A_g = A_high if A_high is not None else A
        H_local = self.local_encoder(X, A)
        H_global = self.global_encoder(X, A_g)
        logits = self.classifier(H_local)
        return {
            "logits": logits,
            "H_local": H_local,
            "H_global": H_global
        }
