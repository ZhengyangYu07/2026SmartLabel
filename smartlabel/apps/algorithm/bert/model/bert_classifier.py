"""BERT 分类模型定义"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class BertTextClassifier(nn.Module):
    """BERT 文本分类模型"""
    
    def __init__(self, bert_model_name, num_classes, hidden_dim=256, dropout_rate=0.3):
        """
        初始化 BERT 分类模型。
        
        Args:
            bert_model_name: 预训练模型名称 (e.g., 'bert-base-chinese')
            num_classes: 分类类别数
            hidden_dim: 隐层维度
            dropout_rate: Dropout 比率
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.bert_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播。
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            token_type_ids: token 类型 IDs（可选）
        
        Returns:
            logits: 分类 logits
        """
        # BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用 [CLS] token 的输出作为句子表示
        pooled_output = outputs.pooler_output
        
        # 两层全连接
        hidden = F.relu(self.fc1(self.dropout(pooled_output)))
        logits = self.fc2(self.dropout(hidden))
        
        return logits
