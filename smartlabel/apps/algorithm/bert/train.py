"""BERT 文本分类训练入口"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import csv
from pathlib import Path

from utils.data_utils import load_text_dataset, encode_labels, get_mask
from model.bert_classifier import BertTextClassifier
from model.text_dataset import TextDataset


def emit_progress(progress):
    """发送进度信息"""
    progress = max(0, min(int(progress), 99))
    print(f"处理进度: {progress}%")


def train(config):
    """主训练函数"""
    print("[Info] 开始 BERT 文本分类训练...")
    
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Info] 使用设备: {device}")
    emit_progress(5)
    
    # 加载数据
    input_path = config.get('text_input_dir') or config.get('dataset_path')
    if not input_path:
        raise ValueError('配置文件中缺少 text_input_dir 或 dataset_path')
    
    labeled_items, unlabeled_items = load_text_dataset(input_path, split=True)
    print(f"[Info] 加载数据: 有标签={len(labeled_items)}, 无标签={len(unlabeled_items)}")
    emit_progress(10)
    
    # 编码标签
    all_items = labeled_items + unlabeled_items
    raw_labels = [item.get('label', '') for item in all_items]
    label_choices = config.get('label_choices', [])
    encoded_labels, label2id, id2label = encode_labels(raw_labels, label_choices)
    mask = get_mask(encoded_labels)
    
    print(f"[Info] 标签映射: {label2id}")
    emit_progress(15)
    
    # 初始化 tokenizer 和模型
    bert_model_name = config.get('bert_model', 'bert-base-chinese')
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = BertTextClassifier(
        bert_model_name,
        num_classes=len(label2id),
        hidden_dim=config.get('hidden_dim', 256),
        dropout_rate=0.3
    ).to(device)
    print(f"[Info] 加载模型: {bert_model_name}")
    emit_progress(20)
    
    # 训练
    if labeled_items:
        print("[Info] 开始训练...")
        
        # 准备训练数据
        texts = [item['text'] for item in labeled_items]
        labels = [item['label'] for item in labeled_items]
        
        dataset = TextDataset(texts, labels, tokenizer, max_length=config.get('max_length', 128), label2id=label2id)
        batch_size = config.get('batch_size', 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        learning_rate = float(config.get('learning_rate', 2e-5))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        epochs = config.get('epochs', 6)
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels_batch = batch['label'].to(device)
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            progress = 20 + (epoch + 1) / epochs * 60
            print(f"[Info] Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")
            emit_progress(progress)
        
        print("[Info] 训练完成")
    else:
        print("[Warn] 没有有标签数据，跳过训练")
        emit_progress(80)
    
    # 推理
    print("[Info] 开始推理...")
    model.eval()
    result_rows = []
    
    with torch.no_grad():
        for idx, item in enumerate(all_items, start=1):
            text = item['text']
            
            encoding = tokenizer(
                text,
                max_length=config.get('max_length', 128),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = F.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())
            pred_label = id2label[pred_idx]
            confidence = float(probs[pred_idx].item())
            
            result_rows.append({
                'text_id': str(idx),
                'content': text,
                'label': pred_label,
                'confidence': confidence,
            })
    
    print(f"[Info] 推理完成: {len(result_rows)} 条")
    emit_progress(85)
    
    # 保存结果
    output_path = Path(config.get('output_path') or Path.cwd())
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / 'result.csv'
    
    with open(result_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text_id', 'content', 'label', 'confidence'])
        writer.writeheader()
        writer.writerows(result_rows)
    
    print(f"[Info] 结果保存到: {result_file}")
    emit_progress(100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='配置文件路径')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    train(config)

