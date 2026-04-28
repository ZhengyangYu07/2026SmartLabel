
import torch
import torch.nn as nn
import torch.nn.functional as F

#图矩阵乘法（自动选择稀疏矩阵乘法或稠密矩阵乘法）
##输入：邻接矩阵A和节点特征矩阵X
##输出：矩阵乘积的结果（邻居特征聚合结果）
def graph_matmul(A, X):
    return torch.sparse.mm(A, X) if A.is_sparse else torch.matmul(A, X)


class GCNLayer(nn.Module):
    #初始化函数
    def __init__(self, in_features, out_features, activation=F.relu, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation
    #前向传播，先邻居聚合，再进行线性变换，最后进行非线性激活
    def forward(self, X, A_hat):
        X = graph_matmul(A_hat, X)
        X = self.fc(X)
        return self.activation(X) if self.activation else X


class CG3Model(nn.Module):
    #初始化函数（输入维度，隐藏层维度，输出维度，分类类别）
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_classes=4):
        super(CG3Model, self).__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim, activation=F.relu)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, activation=None)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, X, A_hat):
        H = self.gcn1(X, A_hat)
        H = self.gcn2(H, A_hat)
        logits = self.classifier(H)
        return H, logits

    def get_loss(self, H_raw, logits, A, labels, mask, alpha=1.0):
        device = H_raw.device
        sup_loss = F.cross_entropy(logits[mask], labels[mask]) if mask.sum(
        ) > 0 else torch.tensor(0.0, device=device)
        return sup_loss
