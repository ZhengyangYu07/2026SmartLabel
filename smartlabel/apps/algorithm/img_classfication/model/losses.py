
"""损失函数集合：局部/全局视图的半监督对比损失与结构生成（边预测）损失，
供 CG3 双路 GCN 训练时对齐两路特征并保持图结构。"""

import torch
import torch.nn.functional as F

def contrastive_loss(H1, H2, labels, mask, temperature=0.5, max_samples=2000):
    """
    Semi-supervised contrastive loss between H1 (local) and H2 (global) views.
    以有标签节点为锚，拉近同类别两视图相似度、推远异类相似度（InfoNCE 形式）。
    """
    device = H1.device
    if H1.size(0) > max_samples:  # 控制计算量，只取前若干样本
        H1 = H1[:max_samples]
        H2 = H2[:max_samples]
        labels = labels[:max_samples]
        mask = mask[:max_samples]
    N = H1.size(0)
    H1 = F.normalize(H1, dim=1)  # 归一化后内积即余弦相似度
    H2 = F.normalize(H2, dim=1)
    sims = torch.matmul(H1, H2.T) / temperature  # [N, N]，温度缩放锐化分布
    exp_sims = torch.exp(sims)

    loss = 0.0
    count = 0

    for i in range(N):
        if mask[i]:  # labeled node
            pos_mask = (labels == labels[i]) & (torch.arange(N).to(device) != i)  # 同类且非自身为正样本
            if pos_mask.sum() == 0:
                continue  # 无同类的锚点跳过
            pos = exp_sims[i][pos_mask].sum()
            neg = exp_sims[i][~pos_mask].sum() + 1e-6
            loss += -torch.log(pos / (pos + neg))  # 正样本占比越大损失越小
            count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=device)


def structure_generation_loss(H, A, num_samples=50000):
    """
    Efficient structure generation loss:
    - Sample positive edges from sparse A
    - Sample equal number of negative edges
    - Predict edge existence via dot product
    用节点特征点积预测图上边是否存在（正边来自 A，负边随机采样），以 BCE 监督。
    """
    device = H.device
    if A.is_sparse:
        A = A.coalesce()
        indices = A.indices().T  # shape: [num_edges, 2]
    else:
        raise ValueError("A must be sparse COO")

    num_edges = indices.size(0)
    sample_size = min(num_edges, num_samples)

    # Sample positive edges
    perm = torch.randperm(num_edges)[:sample_size]
    pos_pairs = indices[perm]
    pos_scores = (H[pos_pairs[:, 0]] * H[pos_pairs[:, 1]]).sum(dim=1)  # 点积作为边存在性打分
    pos_labels = torch.ones(sample_size, device=device)

    # Sample negative edges (random pairs)
    N = H.size(0)
    neg_i = torch.randint(0, N, (sample_size,), device=device)
    neg_j = torch.randint(0, N, (sample_size,), device=device)
    neg_scores = (H[neg_i] * H[neg_j]).sum(dim=1)
    neg_labels = torch.zeros(sample_size, device=device)

    # Combine
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])

    return F.binary_cross_entropy_with_logits(scores, labels)

    H = F.normalize(H, dim=1)
    scores = torch.matmul(H, H.T)
    loss = F.binary_cross_entropy_with_logits(scores, masked_A)
    return loss
