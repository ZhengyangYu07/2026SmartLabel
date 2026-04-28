import torch
import torch.nn as nn
import torch.nn.functional as F

def graph_matmul(A, X):
    return torch.sparse.mm(A, X.to(A.device)) if A.is_sparse else torch.matmul(A, X.to(A.device))

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=F.relu, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, X, A):
        X = graph_matmul(A, X)
        X = self.fc(X)
        return self.activation(X) if self.activation else X

class CG3_Paper(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_classes=4):
        super(CG3_Paper, self).__init__()
        # Local view encoder
        self.local_gcn1 = GCNLayer(in_dim, hidden_dim, activation=F.relu)
        self.local_gcn2 = GCNLayer(hidden_dim, out_dim, activation=None)

        # Global view encoder
        self.global_gcn1 = GCNLayer(in_dim, hidden_dim, activation=F.relu)
        self.global_gcn2 = GCNLayer(hidden_dim, out_dim, activation=None)

        # Classifier
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, X, A, A_high=None):
        # Local view
        H_local = self.local_gcn1(X, A)
        H_local = self.local_gcn2(H_local, A)

        # Global view
        A_g = A_high if A_high is not None else A  # default to A if not given
        H_global = self.global_gcn1(X, A_g)
        H_global = self.global_gcn2(H_global, A_g)

        # Final representation (can be used individually or combined)
        logits = self.classifier(H_local)
        return {
            "logits": logits,
            "H_local": H_local,
            "H_global": H_global
        }
