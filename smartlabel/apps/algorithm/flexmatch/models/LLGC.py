"""LLGC 标签传播。

在图上把有标签样本的标签平滑扩散到无标签样本，直接输出每个节点的类概率分布，
是四路分类器中的无参数图方法。
"""
import torch


def normalize_adjacency(A):
    """对称归一化邻接矩阵 D^{-1/2} A D^{-1/2}，抑制高度数节点的影响。"""
    if A.is_sparse:
        deg = torch.sparse.sum(A, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        # 孤立节点度数为 0，-0.5 次幂会溢出为 inf，统一置 0
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        indices = A._indices()
        values = A._values()
        row, col = indices[0, :], indices[1, :]
        norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(indices, norm_values, A.size(), device=A.device).coalesce()
    else:
        D = torch.diag(torch.sum(A, dim=1))
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0.0
        return D_inv_sqrt @ A @ D_inv_sqrt


def llgc_predict(A, labels, mask, num_classes=4, alpha=0.99, max_iter=100, tol=1e-5):
    """迭代式标签传播 F ← αSF + (1-α)Y，收敛后返回各节点类概率。"""
    N = A.size(0)
    device = A.device

    # Y 为 one-hot 标签指示矩阵，仅标注样本有非零行
    Y = torch.zeros(N, num_classes, device=device)
    for i in range(N):
        if mask[i]:
            Y[i, labels[i]] = 1.0

    S = normalize_adjacency(A)
    F = Y.clone()

    for _ in range(max_iter):
        # 每轮传播邻居标签并锚定回原始标签，α 控制扩散强度
        F_new = alpha * torch.sparse.mm(S, F) + (1 - alpha) * Y
        delta = torch.norm(F_new - F, p='fro')
        F = F_new
        # 变化量小于阈值即认为收敛
        if delta < tol:
            break

    return F
