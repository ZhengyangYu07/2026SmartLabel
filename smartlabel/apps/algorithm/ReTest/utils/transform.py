from typing import Callable
from torchvision import transforms # type: ignore
from PIL import Image
import torch


def train_transform() -> Callable[[Image.Image], torch.Tensor]:
    """
    Customized image transformation pipeline for the dataset.
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

def eval_transform() -> Callable[[Image.Image], torch.Tensor]:
    """
    Customized image transformation pipeline for the dataset.
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
