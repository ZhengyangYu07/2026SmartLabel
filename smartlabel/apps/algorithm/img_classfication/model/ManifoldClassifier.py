
"""流形正则分类器：在 MLP 交叉熵基础上加入图拉普拉斯平滑项，
让图上相邻样本的预测趋于一致，是最终多数投票融合中的一路。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManifoldClassifier(nn.Module):
    """MLP 分类器 + 流形（拉普拉斯）正则，鼓励相连样本 logits 接近。"""
    def __init__(self, in_dim, num_classes, hidden_dims=[128], lambda_lap=0.1):
        super().__init__()
        self.lambda_lap = lambda_lap  # 拉普拉斯正则项权重
        layers = []
        dims = [in_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, X, A=None):
        """前向只需特征 X，图 A 仅在计算损失时用于拉普拉斯正则。"""
        return self.classifier(X)

    def get_loss(self, logits, labels, mask, A=None):
        """交叉熵（仅用有标签样本）+ λ·图拉普拉斯平滑项，返回总损失。"""
        loss_ce = F.cross_entropy(logits[mask], labels[mask])
        loss_lap = 0.0
        if A is not None and A.is_sparse:
            A = A.coalesce()
            edges = A.indices().T
            num_sample = min(10000, edges.size(0))
            idx = torch.randperm(edges.size(0), device=logits.device)[
                :num_sample]
            sampled_edges = edges[idx]  # 采样部分边近似整图正则，避免全图计算过大
            diff = logits[sampled_edges[:, 0]] - logits[sampled_edges[:, 1]]
            loss_lap = (diff ** 2).sum(dim=1).mean()  # 边上两端 logits 的平方差，越小越平滑
        return loss_ce + self.lambda_lap * loss_lap


class ManifoldClassifierWithDropout(ManifoldClassifier):
    """带输入 dropout 的变体，前向时先对特征做随机失活。"""
    def __init__(self, in_dim, num_classes, lambda_lap=0.1, dropout_rate=0.5, config=None):
        if config:
            dropout_rate = config.get("dropout_rate", dropout_rate)
        super().__init__(in_dim, num_classes, lambda_lap, config)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, W):
        """先 dropout 再走父类的前向分类。"""
        x = self.dropout(x)
        return super().forward(x, W)
