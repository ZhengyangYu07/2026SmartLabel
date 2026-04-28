
import torch
import torch.nn as nn
import torch.nn.functional as F


class ManifoldClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dims=[128], lambda_lap=0.1):
        super().__init__()
        self.lambda_lap = lambda_lap
        layers = []
        dims = [in_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, X, A=None):
        return self.classifier(X)

    def get_loss(self, logits, labels, mask, A=None):
        loss_ce = F.cross_entropy(logits[mask], labels[mask])
        loss_lap = 0.0
        if A is not None and A.is_sparse:
            A = A.coalesce()
            edges = A.indices().T
            num_sample = min(10000, edges.size(0))
            idx = torch.randperm(edges.size(0), device=logits.device)[
                :num_sample]
            sampled_edges = edges[idx]
            diff = logits[sampled_edges[:, 0]] - logits[sampled_edges[:, 1]]
            loss_lap = (diff ** 2).sum(dim=1).mean()
        return loss_ce + self.lambda_lap * loss_lap


class ManifoldClassifierWithDropout(ManifoldClassifier):
    def __init__(self, in_dim, num_classes, lambda_lap=0.1, dropout_rate=0.5, config=None):
        if config:
            dropout_rate = config.get("dropout_rate", dropout_rate)
        super().__init__(in_dim, num_classes, lambda_lap, config)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, W):
        x = self.dropout(x)
        return super().forward(x, W)
