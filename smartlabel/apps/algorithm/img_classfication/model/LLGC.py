
"""LLGC 标签传播：在对称归一化的图上迭代传播标签，
使未标注节点从邻居处获得类别分布，是最终多数投票融合中的一路。"""

import torch

# def normalize_adjacency(A):
#     if A.is_sparse:
#         deg = torch.sparse.sum(A, dim=1).to_dense()
#         deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
#         indices = A._indices()
#         values = A._values()
#         row, col = indices[0, :], indices[1, :]
#         norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
#         return torch.sparse_coo_tensor(indices, norm_values, A.size()).coalesce()
#     else:
#         D = torch.diag(torch.sum(A, dim=1))
#         D_inv_sqrt = torch.pow(D, -0.5)
#         D_inv_sqrt[D_inv_sqrt == float('inf')] = 0.0
#         return D_inv_sqrt @ A @ D_inv_sqrt

# def llgc_predict(A, labels, mask, num_classes=4, alpha=0.99):
#     N = A.size(0)
#     device = A.device

#     Y = torch.zeros(N, num_classes).to(device)
#     for i in range(N):
#         if mask[i]:
#             Y[i, labels[i]] = 1.0

#     S = normalize_adjacency(A).to(device)
#     F = Y.clone()

#     for _ in range(50):  # fixed-point iteration
#         F = alpha * torch.sparse.mm(S, F) + (1 - alpha) * Y

#     return F

#对邻接矩阵进行对称归一化，用于卷积操作
def normalize_adjacency(A):
    """对称归一化邻接矩阵 D^{-1/2} A D^{-1/2}（稀疏/稠密两种实现），抑制高度节点的影响。"""
    if A.is_sparse:
        deg = torch.sparse.sum(A, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 孤立点度为 0，置零避免 inf
        indices = A._indices()
        values = A._values()
        row, col = indices[0, :], indices[1, :]
        norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]  # 对每条边做两端度归一化
        return torch.sparse_coo_tensor(indices, norm_values, A.size()).coalesce()
    else:
        deg = torch.sum(A, dim=1)  # 修改：改为先求度向量，不直接对矩阵操作
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)         # 修改：构造对角矩阵
        return D_inv_sqrt @ A @ D_inv_sqrt

#执行LLGC标签传播算法，预测图中所有节点的标签
def llgc_predict(A, labels, mask, num_classes=4, alpha=0.8, max_iter=30, tol=1e-4):  # 修改：调整alpha，迭代次数，添加终止阈值
    """迭代式标签传播：F ← α·S·F + (1-α)·Y，收敛或达最大迭代后返回各节点类别分布。"""
    N = A.size(0)
    device = A.device

    Y = torch.zeros(N, num_classes).to(device)
    for i in range(N):
        if mask[i]:
            Y[i, labels[i]] = 1.0  # 有标签节点用 one-hot 作为固定锚点
        else:
            Y[i] = torch.ones(num_classes).to(device) * 1e-6  # 修改：未标注点用小均匀初始化，避免全零

    S = normalize_adjacency(A).to(device)
    F = Y.clone()

    for _ in range(max_iter):  # 修改：动态迭代，添加收敛判定
        F_old = F.clone()
        F = alpha * torch.sparse.mm(S, F) + (1 - alpha) * Y  # 邻居传播项 + 锚点回拉项
        if torch.norm(F - F_old) < tol:  # 两次迭代变化足够小则提前收敛
            break

    return F
