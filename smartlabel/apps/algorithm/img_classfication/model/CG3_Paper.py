"""CG3 双路图神经网络分类器：用局部 KNN 图与全局扩散图两条 GCN 路径分别提取特征，
再由局部特征分类输出 logits，是最终多数投票融合中的一路。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def graph_matmul(A, X):
    """图聚合：稀疏邻接矩阵用稀疏乘法，稠密矩阵用普通乘法，统一封装。"""
    return torch.sparse.mm(A, X.to(A.device)) if A.is_sparse else torch.matmul(A, X.to(A.device))


class GCNLayer(nn.Module):
    """单层图卷积：先按邻接矩阵聚合邻居特征，再线性变换并激活。"""
    def __init__(self, in_features, out_features, activation=F.relu, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, X, A):
        X = graph_matmul(A, X)  # 邻居特征聚合
        X = self.fc(X)
        return self.activation(X) if self.activation else X

#多层GCN堆栈（堆叠多个GCN层构建深层网络）
class GCNStack(nn.Module):
    """多层 GCN 堆叠，最后一层不加激活以保留线性输出。"""
    def __init__(self, dims, activation=F.relu):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            act = activation if i < len(dims) - 2 else None  # 仅中间层带激活
            self.layers.append(GCNLayer(dims[i], dims[i + 1], activation=act))

    def forward(self, X, A):
        for layer in self.layers:
            X = layer(X, A)
        return X


class CG3_Paper(nn.Module):
    """双路 GCN 主体：local_encoder 走局部图，global_encoder 走全局图，分类头接局部特征。"""
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_classes=4, num_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.local_encoder = GCNStack(dims)
        self.global_encoder = GCNStack(dims)
        self.classifier = nn.Linear(out_dim, num_classes)

    #双路径GCN分别学习局部信息和全局信息
    def forward(self, X, A, A_high=None):
        """两路编码：局部图用 A、全局图用 A_high（缺省回退到 A），返回 logits 与两路特征。"""
        A_g = A_high if A_high is not None else A
        H_local = self.local_encoder(X, A)
        H_global = self.global_encoder(X, A_g)
        logits = self.classifier(H_local)  # 用局部特征做分类
        return {
            "logits": logits,
            "H_local": H_local,
            "H_global": H_global
        }
