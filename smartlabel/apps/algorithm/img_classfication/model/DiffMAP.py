
import torch
import torch.nn as nn
import torch.nn.functional as F

def graph_matmul(A, X):
    return torch.sparse.mm(A, X) if A.is_sparse else torch.matmul(A, X)

class DiffusionMapModel(nn.Module):
    def __init__(self, in_dim, num_classes, use_graph=False):
        super(DiffusionMapModel, self).__init__()
        self.use_graph = use_graph
        self.gc1 = nn.Linear(in_dim, 64)
        self.gc2 = nn.Linear(64, num_classes)

    def forward(self, X, A=None):
        if self.use_graph and A is not None:
            X = graph_matmul(A, X)
        x = F.relu(self.gc1(X))
        logits = self.gc2(x)
        return logits

    def get_loss(self, logits, labels, mask):
        return F.cross_entropy(logits[mask], labels[mask])
