
"""扩散图上的简易图分类器：可选先做一次图聚合，再用两层 MLP 输出类别 logits，
并提供仅在有标签样本上计算的交叉熵损失。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def graph_matmul(A, X):
    """图聚合：稀疏邻接矩阵用稀疏乘法，稠密矩阵用普通乘法。"""
    return torch.sparse.mm(A, X) if A.is_sparse else torch.matmul(A, X)

class DiffusionMapModel(nn.Module):
    """两层 MLP 分类器；use_graph=True 时先按邻接矩阵聚合邻居特征再分类。"""
    def __init__(self, in_dim, num_classes, use_graph=False):
        super(DiffusionMapModel, self).__init__()
        self.use_graph = use_graph
        self.gc1 = nn.Linear(in_dim, 64)
        self.gc2 = nn.Linear(64, num_classes)

    def forward(self, X, A=None):
        """可选图聚合后经两层 MLP 得到类别 logits。"""
        if self.use_graph and A is not None:
            X = graph_matmul(A, X)  # 用图结构做一次邻居特征平滑
        x = F.relu(self.gc1(X))
        logits = self.gc2(x)
        return logits

    def get_loss(self, logits, labels, mask):
        """仅对 mask 为 True 的有标签样本计算交叉熵（半监督训练）。"""
        return F.cross_entropy(logits[mask], labels[mask])
