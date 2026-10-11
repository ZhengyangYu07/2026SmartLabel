"""图构建工具。

在 BERT 句向量上构建 kNN 稀疏图，并用 landmark 近似计算扩散映射降维，
为图神经网络与标签传播提供邻接矩阵和低维流形表示。
"""
import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import coo_matrix


def build_knn_graph(X, k=5, sigma=1.0):
    """在特征上构建 kNN 图，边权为高斯核，返回稀疏 COO 邻接矩阵。"""
    X_np = X.detach().cpu().numpy()
    # 多取 1 个邻居，因为返回结果中第 0 个是节点自身
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean').fit(X_np)
    distances, indices = nbrs.kneighbors(X_np)

    rows, cols, weights = [], [], []
    for i in range(X_np.shape[0]):
        for j in range(1, k + 1):  # 从 1 开始，跳过自身
            neighbor_idx = indices[i][j]
            dist = distances[i][j]
            # 高斯核把欧氏距离转成相似度权重
            weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
            rows.append(i)
            cols.append(neighbor_idx)
            weights.append(weight)

    adj = coo_matrix((weights, (rows, cols)), shape=(X_np.shape[0], X_np.shape[0]))
    # 稀疏存储，避免生成 N×N 稠密邻接矩阵
    return torch.sparse_coo_tensor(
        indices=torch.LongTensor(np.vstack([adj.row, adj.col])),
        values=torch.FloatTensor(adj.data),
        size=adj.shape
    )


def compute_diffusion_map_approx(X, n_components=64, sigma=0.5, n_landmarks=1000, kernel='rbf'):
    """用 landmark（Nyström）近似扩散映射，把高维句向量降到低维流形空间。"""
    X_np = X.detach().cpu().numpy()
    N = X_np.shape[0]
    # 随机采样 landmark，降低大规模核矩阵分解的成本
    idx = np.random.choice(N, min(n_landmarks, N), replace=False)
    X_landmarks = X_np[idx]

    # 只计算 landmark×landmark 与 data×landmark 核矩阵
    Wll = pairwise_kernels(X_landmarks, X_landmarks, metric=kernel, gamma=1 / (2 * sigma ** 2))
    Wnl = pairwise_kernels(X_np, X_landmarks, metric=kernel, gamma=1 / (2 * sigma ** 2))

    # 对 landmark 子矩阵做 SVD，再外推全样本低维嵌入（Nyström 近似）
    U, S, _ = randomized_svd(Wll, n_components=n_components, n_iter=4, random_state=42)
    Z = Wnl @ U @ np.diag(1.0 / np.sqrt(S))
    return torch.tensor(Z, dtype=torch.float32, device=X.device)
