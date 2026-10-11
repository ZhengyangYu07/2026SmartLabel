"""流形正则分类器。

在扩散映射得到的低维流形上跑两层 GCN，用 LLGC 软标签蒸馏 + 拉普拉斯平滑正则联合训练，
把标签传播的结构知识迁移进神经网络分类器。
"""
import torch
import torch.nn as nn
import torch.nn.functional as Func


def llgc_penalty(F, A):
    """拉普拉斯平滑惩罚：相邻节点预测差异越大惩罚越高（按边权加权）。"""
    A = A.coalesce()
    row, col = A.indices()
    diff = F[row] - F[col]
    penalty = (diff ** 2).sum(dim=1) * A.values()
    return penalty.mean()


def graph_matmul(A, X):
    """邻接矩阵与特征相乘（图卷积的邻域聚合步骤）。"""
    A = A.coalesce()
    return torch.sparse.mm(A, X)


class GCNLayer(nn.Module):
    """单层图卷积：邻域聚合 → 线性变换 → 激活。"""

    def __init__(self, in_features, out_features, activation=Func.relu, bias=True):
        """初始化线性变换与激活函数。"""
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, X, A):
        """先按邻接矩阵聚合邻居特征，再线性映射并激活。"""
        X = graph_matmul(A, X)
        X = self.fc(X)
        return self.activation(X) if self.activation else X


class ManifoldClassifier(nn.Module):
    """流形正则分类器：两层 GCN + 线性分类头，配拉普拉斯正则训练。"""

    def __init__(self, in_dim, num_classes, lambda_lap=0.1, k=20, sigma=0.5, lr=1e-4):
        """初始化两层 GCN、分类头与流形正则超参数。"""
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.lr = lr
        self.lambda_lap = lambda_lap

        self.gcn1 = GCNLayer(in_dim, 128)
        self.gcn2 = GCNLayer(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, X, A):
        """两层图卷积后输出各类 logits。"""
        Z = self.gcn1(X, A)
        Z = self.gcn2(Z, A)
        return self.classifier(Z)

    def get_loss(self, logits, labels, mask, A=None):
        """有监督损失 = 标注样本交叉熵 + λ 加权流形正则。"""
        loss_ce = Func.cross_entropy(logits[mask], labels[mask])
        F = torch.softmax(logits, dim=1)
        loss_lp = llgc_penalty(F, A)
        return loss_ce + self.lambda_lap * loss_lp

    def get_soft_loss(self, logits, F_soft, A):
        """半监督损失 = 与 LLGC 软标签的 KL 蒸馏 + 流形正则。"""
        F_pred = torch.softmax(logits, dim=1)
        log_probs = Func.log_softmax(logits, dim=1)
        # 用标签传播得到的软分布 F_soft 作教师信号
        loss_ce = Func.kl_div(log_probs, F_soft, reduction='batchmean')
        loss_lp = llgc_penalty(F_pred, A)
        return loss_ce + self.lambda_lap * loss_lp
