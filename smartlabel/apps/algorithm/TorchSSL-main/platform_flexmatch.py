"""
Platform entry for running FlexMatch on uploaded text data.

Provides a helper `train_from_text_arrays` which accepts numpy/torch Arrays or lists
for labeled/unlabeled/eval data and runs FlexMatch through the existing adapter.

This wrapper keeps changes minimal and relies on `flexmatch_adapter.FlexMatchAdapter`.
"""
from typing import Optional
import torch
from torch.utils.data import DataLoader

from flexmatch_adapter import FlexMatchAdapter
from datasets.text_dataset import TextBasicDataset


def make_loader_from_arrays(data, targets=None, batch_size=32, is_ulb=False, transform=None, strong_transform=None, shuffle=True, num_workers=0):
    dset = TextBasicDataset(data, targets, transform=transform, is_ulb=is_ulb, strong_transform=strong_transform)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dset, loader


def train_from_text_arrays(model,
                           num_classes: int,
                           labeled_data,
                           labeled_targets,
                           unlabeled_data,
                           eval_data,
                           eval_targets,
                           optimizer,
                           scheduler,
                           args,
                           device: str = 'cuda',
                           batch_size: int = 32,
                           num_workers: int = 0,
                           transform=None,
                           strong_transform=None):
    """Train FlexMatch on text arrays.

    Args:
        model: nn.Module instance (unwrapped)
        num_classes: int
        labeled_data: list/array of labeled inputs
        labeled_targets: list/array of labels
        unlabeled_data: list/array of unlabeled inputs
        eval_data, eval_targets: evaluation set
        optimizer: torch optimizer
        scheduler: lr scheduler or None
        args: argparse.Namespace or object with required fields used by FlexMatch (epoch, num_train_iter, etc.)
        device: 'cuda' or 'cpu'
        batch_size, num_workers: DataLoader params
        transform, strong_transform: optional callables

    Returns:
        evaluation dict from FlexMatch training
    """

    # prepare loaders
    lb_dset, lb_loader = make_loader_from_arrays(labeled_data, labeled_targets, batch_size=batch_size, is_ulb=False, transform=transform, num_workers=num_workers, shuffle=True)
    ulb_dset, ulb_loader = make_loader_from_arrays(unlabeled_data, None, batch_size=batch_size, is_ulb=True, transform=transform, strong_transform=strong_transform, num_workers=num_workers, shuffle=True)
    eval_dset, eval_loader = make_loader_from_arrays(eval_data, eval_targets, batch_size=batch_size, is_ulb=False, transform=transform, num_workers=num_workers, shuffle=False)

    adapter = FlexMatchAdapter(model=model, num_classes=num_classes, device=device)
    adapter.set_unlabeled_dataset(ulb_dset)
    adapter.set_data_loaders(lb_loader, ulb_loader, eval_loader)
    adapter.set_optimizer(optimizer, scheduler)

    # run training
    result = adapter.train(args)
    return result
