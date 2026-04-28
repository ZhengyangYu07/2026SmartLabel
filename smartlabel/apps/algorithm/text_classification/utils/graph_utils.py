import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import coo_matrix

def build_knn_graph(X, k=5, sigma=1.0):
    X_np = X.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean').fit(X_np)
    distances, indices = nbrs.kneighbors(X_np)

    rows, cols, weights = [], [], []
    for i in range(X_np.shape[0]):
        for j in range(1, k + 1):
            neighbor_idx = indices[i][j]
            dist = distances[i][j]
            weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
            rows.append(i)
            cols.append(neighbor_idx)
            weights.append(weight)

    adj = coo_matrix((weights, (rows, cols)), shape=(X_np.shape[0], X_np.shape[0]))
    return torch.sparse_coo_tensor(
        indices=torch.LongTensor(np.vstack([adj.row, adj.col])),
        values=torch.FloatTensor(adj.data),
        size=adj.shape
    )

def compute_diffusion_map_approx(X, n_components=64, sigma=0.5, n_landmarks=1000, kernel='rbf'):
    X_np = X.detach().cpu().numpy()
    N = X_np.shape[0]
    idx = np.random.choice(N, min(n_landmarks, N), replace=False)
    X_landmarks = X_np[idx]

    Wll = pairwise_kernels(X_landmarks, X_landmarks, metric=kernel, gamma=1 / (2 * sigma ** 2))
    Wnl = pairwise_kernels(X_np, X_landmarks, metric=kernel, gamma=1 / (2 * sigma ** 2))

    U, S, _ = randomized_svd(Wll, n_components=n_components, n_iter=4, random_state=42)
    Z = Wnl @ U @ np.diag(1.0 / np.sqrt(S))
    return torch.tensor(Z, dtype=torch.float32, device=X.device)
