"""BERT 文本数据集定义"""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """BERT 文本数据集"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128, label2id=None):
        """
        初始化数据集。
        
        Args:
            texts: 文本列表
            labels: 标签列表 (可以是字符串或数字 ID)
            tokenizer: BERT tokenizer
            max_length: 最大文本长度
            label2id: 标签到 ID 的映射 (如果 labels 是字符串)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 如果提供了 label2id 映射，转换字符串标签为 ID
        if self.label2id is not None and isinstance(label, str):
            label = self.label2id[label]
        
        # Tokenize 文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
