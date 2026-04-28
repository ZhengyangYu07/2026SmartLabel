import os
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path


def scan_image_directory(image_dir: str) -> List[str]:
    """
    扫描图片目录，返回所有图片文件路径

    Args:
        image_dir: 图片目录路径

    Returns:
        List[str]: 图片文件路径列表
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []

    # 确保路径存在
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    for root, _, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    image_paths.sort()  # 确保顺序一致
    print(f"[DataUtils] 在 {image_dir} 中找到 {len(image_paths)} 张图片")
    return image_paths


def load_label_csv(csv_path: str, image_paths: List[str]) -> Dict[str, int]:
    """
    加载标签CSV文件，建立文件路径到标签的映射

    Args:
        csv_path: CSV文件路径
        image_paths: 所有图片路径列表

    Returns:
        Dict[str, int]: {文件路径: 标签} 的字典
    """
    if not os.path.exists(csv_path):
        print(f"[DataUtils] 标签文件不存在: {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
        print(f"[DataUtils] 成功加载标签文件: {csv_path}")
        print(f"[DataUtils] 标签文件列名: {df.columns.tolist()}")
    except Exception as e:
        print(f"[DataUtils] 加载标签文件失败: {e}")
        return {}

    # 检查必要列 - 更灵活的列名匹配
    filename_col = None
    label_col = None

    for col in df.columns:
        if 'filename' in col.lower() or 'name' in col.lower():
            filename_col = col
        if 'class' in col.lower() or 'label' in col.lower():
            label_col = col

    if not filename_col or not label_col:
        available_cols = df.columns.tolist()
        raise ValueError(f"CSV文件必须包含文件名和标签列。可用列: {available_cols}")

    print(f"[DataUtils] 使用文件名列: {filename_col}, 标签列: {label_col}")

    # 创建文件名到路径的映射（支持多种文件名格式）
    file_to_path = {}
    for path in image_paths:
        filename = os.path.basename(path)
        file_to_path[filename] = path
        # 也支持不带扩展名的匹配
        filename_no_ext = os.path.splitext(filename)[0]
        file_to_path[filename_no_ext] = path

    # 构建标签字典
    label_dict = {}
    missing_files = []

    for _, row in df.iterrows():
        filename = str(row[filename_col]).strip()

        if filename in file_to_path:
            label_dict[file_to_path[filename]] = int(row[label_col])
        else:
            missing_files.append(filename)

    if missing_files:
        print(f"[DataUtils] 警告: {len(missing_files)} 个标签文件中的图片未找到")
        if len(missing_files) <= 5:  # 只显示前5个
            print(f"[DataUtils] 未找到的图片: {missing_files[:5]}")

    print(f"[DataUtils] 成功加载 {len(label_dict)} 个标签")
    return label_dict


def split_labeled_unlabeled(image_paths: List[str], label_dict: Dict[str, int]) -> Tuple[List[str], List[str]]:
    """
    分割已标注和未标注数据

    Args:
        image_paths: 所有图片路径
        label_dict: 标签字典

    Returns:
        Tuple[List[str], List[str]]: (已标注路径列表, 未标注路径列表)
    """
    labeled_paths = []
    unlabeled_paths = []

    for path in image_paths:
        if path in label_dict:
            labeled_paths.append(path)
        else:
            unlabeled_paths.append(path)

    print(f"[DataUtils] 已标注: {len(labeled_paths)}, 未标注: {len(unlabeled_paths)}")
    return labeled_paths, unlabeled_paths


def inspect_data_structure(base_path: str):
    """
    检查数据目录结构，帮助调试

    Args:
        base_path: 基础路径
    """
    print(f"\n[DataUtils] 检查数据目录结构: {base_path}")

    if not os.path.exists(base_path):
        print(f"  目录不存在: {base_path}")
        return

    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 2 * (level + 1)

        # 显示前10个文件
        for file in files[:10]:
            print(f'{sub_indent}{file}')
        if len(files) > 10:
            print(f'{sub_indent}... 还有 {len(files) - 10} 个文件')