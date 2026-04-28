import torch
from typing import Any, Tuple

class QueueFeatureLabels:
    def __init__(self, cfg: Any, device: torch.device):
        self.K: int= cfg.queue_per_class * cfg.n_classes * 2
        self.features: torch.Tensor= -1.0 * torch.ones(self.K, cfg.low_dim).to(device)
        self.labels: torch.Tensor= -1.0 * torch.ones(self.K, dtype=torch.long).to(device)
        self.indices: torch.Tensor= -1.0 * torch.ones(self.K, dtype=torch.long).to(device)
        self.ptr: int= 0

    @property
    def is_full(self) -> bool:
        return self.indices[-1].item() != -1

    def get_features_labels_indices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_full:
            return self.features, self.labels, self.indices
        else:
            return self.features[:self.ptr], self.labels[:self.ptr], self.indices[:self.ptr]

    # enqueue features & probabilities & indices
    def enqueue_dequeue(self, features: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> None:
        q_size = labels.size(0)

        if self.ptr + q_size > self.K:
            self.features[-q_size:] = features
            self.labels[-q_size:] = labels
            self.indices[-q_size:] = torch.cat([indices, indices], dim = 0)
            self.ptr = 0
        else:
            self.features[self.ptr: self.ptr + q_size] = features
            self.labels[self.ptr: self.ptr + q_size] = labels
            self.indices[self.ptr: self.ptr + q_size] = torch.cat([indices, indices], dim = 0)
            
            self.ptr += q_size