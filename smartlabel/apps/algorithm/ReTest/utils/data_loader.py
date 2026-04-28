from typing import Tuple, Dict, Any, Optional, Callable
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # type: ignore
from PIL import Image


class CustomImageDataset(Dataset[Tuple[torch.Tensor, int, str, str]]):
    """
    A custom dataset that loads image paths and noisy labels from a CSV file.

    The dataset expects a config dictionary with the following fields:
        - csv_path:          path to the CSV file containing data
        - image_path_column: column name containing image paths
        - label_column:      column name containing noisy labels
    """

    def __init__(self, config: Dict[str, Any], transform: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        self.csv_path: str = config.get("csv_path", "data/result.csv")
        self.label_col: str = config.get("label_column", "label")
        self.image_path_col: str = config.get("image_path_column", "image_path")
        self.image_name_col: str = config.get("image_name_column", "image_name")
        self.output_path: str = config.get("output_path", "output/final_result.csv")

        self.data: pd.DataFrame = pd.read_csv(self.csv_path) # pyright: ignore[reportUnknownMemberType]

        # Define image preprocessing pipeline
        self.transform: Callable[[Image.Image], torch.Tensor] = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        # Build label-to-index and index-to-label mappings
        labels = self.data[self.label_col].unique() # type: ignore
        self.class_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(sorted(labels))} # type: ignore
        self.idx_to_class: Dict[int, str] = {idx: label for label, idx in self.class_to_idx.items()}
        self.num_classes: int = len(self.class_to_idx)

    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        """
        Load an image and its corresponding label index.

        Returns:
            image: the transformed image
            label: the integer label
            name:  the name of the image file
            path:  the path to the image file
        """
        row = self.data.iloc[idx]
        image = Image.open(row[self.image_path_col]).convert("RGB")
        image = self.transform(image)
        label_str: str = row[self.label_col]
        label: int = self.class_to_idx[label_str]
        name: str = row[self.image_name_col]
        path: str = row[self.image_path_col]

        return image, label, name, path


def get_dataloader(config: Dict[str, Any], batch_size: int = 64, shuffle: bool = True, transform: Optional[Callable[[Image.Image], torch.Tensor]] = None) -> DataLoader[Tuple[torch.Tensor, int, str, str]]:
    """
    Initialize a DataLoader from the given config dictionary.

    Args:
        config: config["dataset_config"] dict containing dataset options
        batch_size: number of samples per batch
        shuffle: whether to shuffle the dataset

    Returns:
        PyTorch DataLoader object
    """
    dataset = CustomImageDataset(config, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
