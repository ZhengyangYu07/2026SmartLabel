"""MixText 分类器。

用 mixup 在特征空间对样本及其标签做线性插值，以数据增强方式做半监督训练，
是四路分类器中的增强式 MLP 分支。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixTextClassifier(nn.Module):
    """MixText 主模型：两层 MLP + dropout 的分类头。"""

    def __init__(self, in_dim, num_classes, hidden_dim=512, dropout=0.3):
        """初始化两层全连接与 dropout。"""
        super(MixTextClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """前向得到各类 logits。"""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


def mixup_data(x, y, alpha=1.0):
    """按 Beta 分布采样混合系数，随机配对插值样本与标签。"""
    if alpha > 0:
        # Beta 采样混合比例 λ；α≤0 时退化为不混合
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    # 随机打乱配对并线性插值特征
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # 返回两套标签与 λ，供训练时加权交叉熵
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
