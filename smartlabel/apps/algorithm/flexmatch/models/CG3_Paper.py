"""CG3 图神经网络分类器。

双通道 GCN（局部图/全局图）+ 对比损失 + 结构生成损失，在 kNN 图上做半监督节点分类，
是四路分类器中的图模型主力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def graph_matmul(A, X):
    """邻接矩阵与特征相乘（图卷积的邻域聚合步骤），兼容稀疏与稠密 A。"""
    return torch.sparse.mm(A, X.to(A.device)) if A.is_sparse else torch.matmul(A, X.to(A.device))


class GCNLayer(nn.Module):
    """单层图卷积：邻域聚合 → 线性变换 → 激活。"""

    def __init__(self, in_features, out_features, activation=F.relu, bias=True):
        """初始化线性变换与激活函数。"""
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, X, A):
        """先按邻接矩阵聚合邻居特征，再线性映射并激活。"""
        X = graph_matmul(A, X)
        X = self.fc(X)
        return self.activation(X) if self.activation else X


class CG3_Paper(nn.Module):
    """CG3 主模型：局部/全局两条 GCN 分支 + 共享分类头。"""

    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_classes=4):
        """构建 local/global 各两层 GCN 及分类头。"""
        super(CG3_Paper, self).__init__()
        self.local_gcn1 = GCNLayer(in_dim, hidden_dim, activation=F.relu)
        self.local_gcn2 = GCNLayer(hidden_dim, out_dim, activation=None)
        self.global_gcn1 = GCNLayer(in_dim, hidden_dim, activation=F.relu)
        self.global_gcn2 = GCNLayer(hidden_dim, out_dim, activation=None)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, X, A, A_high=None):
        """前向：局部分支出 logits，全局分支出表示，供对比/结构损失使用。"""
        H_local = self.local_gcn1(X, A)
        H_local = self.local_gcn2(H_local, A)
        # 未提供高层图时，全局分支退化为使用同一张图
        A_g = A_high if A_high is not None else A
        H_global = self.global_gcn1(X, A_g)
        H_global = self.global_gcn2(H_global, A_g)
        # 分类只用局部表示
        logits = self.classifier(H_local)
        return {
            "logits": logits,
            "H_local": H_local,
            "H_global": H_global
        }
