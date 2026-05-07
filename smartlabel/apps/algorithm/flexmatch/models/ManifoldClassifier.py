import torch
import torch.nn as nn
import torch.nn.functional as Func


def llgc_penalty(F, A):
    A = A.coalesce()
    row, col = A.indices()
    diff = F[row] - F[col]
    penalty = (diff ** 2).sum(dim=1) * A.values()
    return penalty.mean()


def graph_matmul(A, X):
    A = A.coalesce()
    return torch.sparse.mm(A, X)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=Func.relu, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, X, A):
        X = graph_matmul(A, X)
        X = self.fc(X)
        return self.activation(X) if self.activation else X


class ManifoldClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, lambda_lap=0.1, k=20, sigma=0.5, lr=1e-4):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.lr = lr
        self.lambda_lap = lambda_lap

        self.gcn1 = GCNLayer(in_dim, 128)
        self.gcn2 = GCNLayer(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, X, A):
        Z = self.gcn1(X, A)
        Z = self.gcn2(Z, A)
        return self.classifier(Z)

    def get_loss(self, logits, labels, mask, A=None):
        loss_ce = Func.cross_entropy(logits[mask], labels[mask])
        F = torch.softmax(logits, dim=1)
        loss_lp = llgc_penalty(F, A)
        return loss_ce + self.lambda_lap * loss_lp

    def get_soft_loss(self, logits, F_soft, A):
        F_pred = torch.softmax(logits, dim=1)
        log_probs = Func.log_softmax(logits, dim=1)
        loss_ce = Func.kl_div(log_probs, F_soft, reduction='batchmean')
        loss_lp = llgc_penalty(F_pred, A)
        return loss_ce + self.lambda_lap * loss_lp
