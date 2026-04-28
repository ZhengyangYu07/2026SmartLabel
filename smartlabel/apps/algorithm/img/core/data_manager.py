import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import os
from PIL import Image
# 临时解决方案 - 直接导入
import sys
import os

# 获取当前文件的绝对路径，然后构建到 utils 的路径
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)  # img 目录
utils_dir = os.path.join(parent_dir, 'utils')

# 添加 utils 目录到 Python 路径
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

# 现在可以直接导入
from data_utils import scan_image_directory, load_label_csv, split_labeled_unlabeled

class SimpleImageDataset(Dataset):
    """
    简单的图像数据集 - 动态加载图片，避免内存爆炸
    """

    def __init__(self, image_paths: List[str], labels: Optional[List[int]] = None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.has_labels = labels is not None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 动态加载图片
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个黑色图片作为fallback
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            return image, self.labels[idx]
        else:
            return image

    def get_path(self, idx: int) -> str:
        """获取图片路径"""
        return self.image_paths[idx]


class SimpleDataManager:
    """
    简化版数据管理器 - 专注于核心数据处理功能
    """

    def __init__(self, image_dir: str, label_csv: Optional[str] = None):
        """
        初始化数据管理器

        Args:
            image_dir: 图片目录
            label_csv: 标签CSV文件（可选）
        """

        # 扫描图片
        self.all_image_paths = scan_image_directory(image_dir)

        # 加载标签
        self.label_dict = load_label_csv(label_csv, self.all_image_paths) if label_csv else {}

        # 分割数据
        self.labeled_paths, self.unlabeled_paths = split_labeled_unlabeled(
            self.all_image_paths, self.label_dict
        )

        # 构建索引映射
        self._build_index_mapping()

        print(f"[DataManager] 初始化完成 - 总计: {len(self.all_image_paths)}, "
              f"已标注: {len(self.labeled_paths)}, 未标注: {len(self.unlabeled_paths)}")

    def _build_index_mapping(self):
        """构建路径到索引的映射"""
        self.path_to_index = {path: idx for idx, path in enumerate(self.all_image_paths)}
        self.labeled_indices = [self.path_to_index[path] for path in self.labeled_paths]
        self.unlabeled_indices = [self.path_to_index[path] for path in self.unlabeled_paths]

    def get_labeled_dataset(self, transform=None) -> SimpleImageDataset:
        """获取已标注数据集"""
        labels = [self.label_dict[path] for path in self.labeled_paths]
        return SimpleImageDataset(self.labeled_paths, labels, transform)

    def get_unlabeled_dataset(self, transform=None) -> SimpleImageDataset:
        """获取未标注数据集"""
        return SimpleImageDataset(self.unlabeled_paths, transform=transform)

    def get_all_dataset(self, transform=None) -> SimpleImageDataset:
        """获取所有数据集"""
        all_labels = [self.label_dict.get(path, -1) for path in self.all_image_paths]
        return SimpleImageDataset(self.all_image_paths, all_labels, transform)

    def add_labels(self, new_labels: Dict[str, int]):
        """添加新标注"""
        for path, label in new_labels.items():
            if path in self.all_image_paths and path not in self.label_dict:
                self.label_dict[path] = label
                # 更新数据分割
                self.labeled_paths, self.unlabeled_paths = self._update_data_split()

    def _update_data_split(self) -> Tuple[List[str], List[str]]:
        """更新数据分割"""
        labeled = []
        unlabeled = []
        for path in self.all_image_paths:
            if path in self.label_dict:
                labeled.append(path)
            else:
                unlabeled.append(path)
        return labeled, unlabeled

    def get_statistics(self) -> Dict:
        """获取数据统计"""
        total = len(self.all_image_paths)
        labeled = len(self.labeled_paths)
        return {
            'total_images': total,
            'labeled_count': labeled,
            'unlabeled_count': total - labeled,
            'label_ratio': labeled / total if total > 0 else 0,
            'completion_percentage': (labeled / total * 100) if total > 0 else 0
        }

    def get_unlabeled_batch(self, batch_size: int = 10) -> List[Tuple[int, str]]:
        """获取一批未标注数据（用于主动学习）"""
        batch_indices = self.unlabeled_indices[:batch_size]
        return [(idx, self.all_image_paths[idx]) for idx in batch_indices]