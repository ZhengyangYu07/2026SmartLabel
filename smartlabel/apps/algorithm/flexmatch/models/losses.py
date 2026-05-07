import torch
import torch.nn.functional as F


def contrastive_loss(H1, H2, labels, mask, temperature=0.5, max_samples=2000):
    device = H1.device
    if H1.size(0) > max_samples:
        H1 = H1[:max_samples]
        H2 = H2[:max_samples]
        labels = labels[:max_samples]
        mask = mask[:max_samples]
    N = H1.size(0)
    H1 = F.normalize(H1, dim=1)
    H2 = F.normalize(H2, dim=1)
    sims = torch.matmul(H1, H2.T) / temperature
    exp_sims = torch.exp(sims)

    loss = 0.0
    count = 0

    for i in range(N):
        if mask[i]:
            pos_mask = (labels == labels[i]) & (torch.arange(N).to(device) != i)
            if pos_mask.sum() == 0:
                continue
            pos = exp_sims[i][pos_mask].sum()
            neg = exp_sims[i][~pos_mask].sum() + 1e-6
            loss += -torch.log(pos / (pos + neg))
            count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=device)


def structure_generation_loss(H, A, num_samples=50000):
    device = H.device
    if A.is_sparse:
        A = A.coalesce()
        indices = A.indices().T
    else:
        raise ValueError("A must be sparse COO")

    num_edges = indices.size(0)
    sample_size = min(num_edges, num_samples)

    perm = torch.randperm(num_edges)[:sample_size]
    pos_pairs = indices[perm]
    pos_scores = (H[pos_pairs[:, 0]] * H[pos_pairs[:, 1]]).sum(dim=1)
    pos_labels = torch.ones(sample_size, device=device)

    N = H.size(0)
    neg_i = torch.randint(0, N, (sample_size,), device=device)
    neg_j = torch.randint(0, N, (sample_size,), device=device)
    neg_scores = (H[neg_i] * H[neg_j]).sum(dim=1)
    neg_labels = torch.zeros(sample_size, device=device)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])

    return F.binary_cross_entropy_with_logits(scores, labels)
