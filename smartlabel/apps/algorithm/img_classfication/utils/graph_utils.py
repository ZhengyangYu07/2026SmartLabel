
import torch
import faiss
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

def build_knn_graph(X, k=5):
    X_np = torch.nn.functional.normalize(X, dim=1).cpu().numpy()
    index = faiss.IndexFlatIP(X_np.shape[1])
    index.add(X_np)
    D, I = index.search(X_np, k + 1)

    rows, cols = [], []
    for i in range(X_np.shape[0]):
        for j in I[i]:
            if i != j:
                rows.append(i)
                cols.append(j)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    A = torch.sparse_coo_tensor(indices, values, size=(X_np.shape[0], X_np.shape[0]))
    return A.coalesce()

def compute_diffusion_map(X, k=10, sigma=1.0):
    """
    基于扩散映射构建邻接矩阵。

    参数：
    - X: torch.Tensor，形状 (N, D)，输入特征
    - k: int，最近邻数
    - sigma: float，高斯核带宽

    返回：
    - A_diffmap: torch.sparse.FloatTensor，形状 (N, N)，对称稀疏邻接矩阵
    """
    X_np = X.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_np)
    distances, indices = nbrs.kneighbors(X_np)

    N = X_np.shape[0]
    rows, cols, vals = [], [], []

    for i in range(N):
        for j in range(k):
            idx = indices[i, j]
            dist = distances[i, j]
            sim = np.exp(-dist ** 2 / (2 * sigma ** 2))
            rows.append(i)
            cols.append(idx)
            vals.append(sim)

    # 构造稀疏矩阵并对称化
    W = csr_matrix((vals, (rows, cols)), shape=(N, N))
    W = (W + W.T) / 2  # 对称化

    # 转换为 PyTorch 稀疏张量
    W_coo = W.tocoo()
    indices = torch.tensor([W_coo.row, W_coo.col], dtype=torch.long)
    values = torch.tensor(W_coo.data, dtype=torch.float32)
    A_diffmap = torch.sparse_coo_tensor(indices, values, size=(N, N))
    return A_diffmap.coalesce()

# def compute_diffusion_map(X, k=10, sigma=1.0, n_components=64):
#     X_np = X.detach().cpu().numpy()
#     nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_np)
#     distances, indices = nbrs.kneighbors(X_np)

#     N = X_np.shape[0]
#     rows, cols, vals = [], [], []

#     for i in range(N):
#         for j in range(k):
#             idx = indices[i, j]
#             dist = distances[i, j]
#             sim = np.exp(-dist ** 2 / (2 * sigma ** 2))
#             rows.append(i)
#             cols.append(idx)
#             vals.append(sim)

#     W = csr_matrix((vals, (rows, cols)), shape=(N, N))
#     W = (W + W.T) / 2  # 对称化

#     D = np.array(W.sum(axis=1)).flatten()
#     D_inv_vec = 1.0 / (D + 1e-6)
#     P = W.multiply(D_inv_vec[:, np.newaxis])  # 稀疏按行缩放：D^{-1}W

#     svd = TruncatedSVD(n_components=n_components + 1)
#     X_diff = svd.fit_transform(P)
#     return torch.tensor(X_diff[:, 1:], dtype=torch.float32)





# def build_knn_graph(X, k=5): 
#     X = torch.nn.functional.normalize(X, dim=1)
#     dtype = X.dtype  # 获取输入特征的 dtype（float32 / float16）
#     X_np = X.detach().cpu().numpy()
    
#     index = faiss.IndexFlatIP(X_np.shape[1])
#     index.add(X_np)
#     D, I = index.search(X_np, k + 1)

#     rows, cols = [], []
#     for i in range(X_np.shape[0]):
#         for j in I[i]:
#             if i != j:
#                 rows.append(i)
#                 cols.append(j)
#     indices = torch.tensor([rows, cols], dtype=torch.long)
#     values = torch.ones(len(rows), torch.float32)  # 保持和 X 一致的 dtype
#     A = torch.sparse_coo_tensor(indices, values, size=(X_np.shape[0], X_np.shape[0]), dtype=torch.float32)
#     return A.coalesce().to(X.device)  # 保持 device 一致

# def compute_diffusion_map(X, k=10, sigma=1.0, n_components=64):
#     dtype = X.dtype  # 保留输入 dtype
#     device = X.device

#     X_np = X.detach().cpu().numpy()
#     nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_np)
#     distances, indices = nbrs.kneighbors(X_np)

#     N = X_np.shape[0]
#     rows, cols, vals = [], [], []

#     for i in range(N):
#         for j in range(k):
#             idx = indices[i, j]
#             dist = distances[i, j]
#             sim = np.exp(-dist ** 2 / (2 * sigma ** 2))
#             rows.append(i)
#             cols.append(idx)
#             vals.append(sim)

#     W = csr_matrix((vals, (rows, cols)), shape=(N, N))
#     W = (W + W.T) / 2  # 对称化

#     D = np.array(W.sum(axis=1)).flatten()
#     D_inv_vec = 1.0 / (D + 1e-6)
#     P = W.multiply(D_inv_vec[:, np.newaxis])

#     svd = TruncatedSVD(n_components=n_components + 1)
#     X_diff = svd.fit_transform(P)

#     return torch.tensor(X_diff[:, 1:], dtype=dtype).to(device)
