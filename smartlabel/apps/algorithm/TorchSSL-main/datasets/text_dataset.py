from torch.utils.data import Dataset
import torch


class TextBasicDataset(Dataset):
    """A minimal dataset wrapper for text data to work with FlexMatch adapter.

    Expected behavior:
    - For labeled dataset: __getitem__ returns (idx, text_tensor, label)
    - For unlabeled dataset: __getitem__ returns (idx, weak_tensor, strong_tensor)
    - For eval dataset: same as labeled dataset

    The dataset stores raw tensors/arrays. If you need tokenization/augmentation,
    pass callable `transform` and `strong_transform` that accept one sample and return a tensor.
    """

    def __init__(self, data, targets=None, transform=None, is_ulb=False, strong_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.is_ulb = is_ulb
        self.strong_transform = strong_transform if strong_transform is not None else transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x_w = self.transform(x)
        else:
            x_w = x
        if not self.is_ulb:
            y = None if self.targets is None else self.targets[idx]
            return idx, x_w, y
        else:
            if self.strong_transform is not None:
                x_s = self.strong_transform(x)
            else:
                x_s = x_w
            return idx, x_w, x_s
