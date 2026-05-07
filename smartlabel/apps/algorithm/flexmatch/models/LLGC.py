import torch


def normalize_adjacency(A):
    if A.is_sparse:
        deg = torch.sparse.sum(A, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
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
    N = A.size(0)
    device = A.device

    Y = torch.zeros(N, num_classes, device=device)
    for i in range(N):
        if mask[i]:
            Y[i, labels[i]] = 1.0

    S = normalize_adjacency(A)
    F = Y.clone()

    for _ in range(max_iter):
        F_new = alpha * torch.sparse.mm(S, F) + (1 - alpha) * Y
        delta = torch.norm(F_new - F, p='fro')
        F = F_new
        if delta < tol:
            break

    return F
