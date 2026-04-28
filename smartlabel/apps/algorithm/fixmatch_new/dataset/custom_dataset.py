import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_csv, label2id, transform=None, test=False):
        self.image_dir = image_dir
        self.df = pd.read_csv(label_csv)
        self.label2id = label2id
        self.transform = transform
        self.test_mode = test  # 测试模式不加载标签

        # 仅加载测试集时不需要标签
        if self.test_mode:
            # 测试集：所有 image_dir 中未出现在 label_csv 的文件
            labeled_filenames = set(self.df["filename"].tolist())
            self.image_files = [f for f in os.listdir(image_dir) if f not in labeled_filenames]
        else:
            # 训练/验证集：读取 CSV 中指定的文件名
            self.image_files = self.df["filename"].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"无法打开图像：{img_path}，错误信息：{e}")

        if self.transform:
            image = self.transform(image)
        
        if self.test_mode:
            return image  # 返回单个 image，避免 test 阶段 unpack 报错
        else:
            # 防止空 label 报错
            match = self.df[self.df["filename"] == img_name]
            if len(match) == 0:
                raise ValueError(f"图像 {img_name} 在标签文件中未找到。")
            label = match["label"].values[0]
            return image, self.label2id[label]  # ✅ 确保是 (image, label)


class UnlabeledDataset(Dataset):
    def __init__(self, image_dir, unlabeled_files, weak_transform, strong_transform):
        self.image_dir = image_dir
        self.unlabeled_files = unlabeled_files  # List[str]
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.unlabeled_files)

    def __getitem__(self, idx):
        img_name = self.unlabeled_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"无法打开图像：{img_path}，错误信息：{e}")

        weak_img = self.weak_transform(image)
        strong_img = self.strong_transform(image)

        return (weak_img, strong_img), -1  # 保持一致：两个 image，加一个 dummy label
