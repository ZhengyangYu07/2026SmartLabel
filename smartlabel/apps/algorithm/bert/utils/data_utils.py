"""BERT 文本分类数据加载和处理工具"""

import pandas as pd
import csv
from pathlib import Path


def load_csv_items(csv_path):
    """加载 CSV 文件中的文本和标签"""
    items = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text') or row.get('content') or ''
            label = row.get('label') or ''
            items.append({'text': text, 'label': label})
    return items


def find_csv_file(input_path):
    """在目录中查找第一个 CSV 文件"""
    path = Path(input_path)
    if path.is_file():
        return path
    csv_files = sorted(path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'未在 {path} 中找到 CSV 文件')
    return csv_files[0]


def load_text_dataset(input_path, split=True):
    """
    加载文本数据集。
    
    Args:
        input_path: CSV 文件路径或包含 CSV 的目录
        split: 是否区分有标签和无标签数据
    
    Returns:
        labeled_list, unlabeled_list 或合并后的列表
    """
    csv_path = find_csv_file(input_path)
    rows = load_csv_items(csv_path)
    
    if split:
        labeled = [row for row in rows if str(row.get('label') or '').strip()]
        unlabeled = [row for row in rows if not str(row.get('label') or '').strip()]
        return labeled, unlabeled
    else:
        return rows


def encode_labels(raw_labels, label_choices=None):
    """
    编码标签为整数 ID，返回映射关系。
    
    Args:
        raw_labels: 原始标签列表
        label_choices: 预定义的标签列表（可选）
    
    Returns:
        (encoded_labels, label2id, id2label)
    """
    # 优先使用预定义标签
    if label_choices:
        labels = [label for label in label_choices if label]
    else:
        # 从数据中提取唯一标签
        labels = sorted({str(label or '').strip() for label in raw_labels if str(label or '').strip()})
    
    # 如果没有标签，使用默认值
    if not labels:
        labels = ['正面', '负面', '中性']
    
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # 编码标签
    encoded = []
    for label in raw_labels:
        label_str = str(label or '').strip()
        if label_str in label2id:
            encoded.append(label2id[label_str])
        else:
            encoded.append(-1)  # 未知标签标记为 -1
    
    return encoded, label2id, id2label


def get_mask(encoded_labels):
    """获取有效标签的掩码（有标签为 True，无标签为 False）"""
    return [label >= 0 for label in encoded_labels]
