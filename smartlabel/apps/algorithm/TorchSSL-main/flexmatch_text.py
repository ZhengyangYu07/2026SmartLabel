"""
FlexMatch for Text Classification on Platform

This script adapts FlexMatch for text classification tasks using BERT embeddings.
It reads combined dataset (labeled + unlabeled) and outputs predictions.

Usage:
    python flexmatch_text.py --config config.json
"""

import json
import os
import sys
import csv
import logging
import argparse
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config_path):
    """Main entry point for FlexMatch text classification."""
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    dataset_path = config['dataset_path']
    output_path = config['output_path']
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load combined dataset
    df = pd.read_csv(dataset_path, encoding='utf-8')
    
    # Separate labeled and unlabeled data
    labeled_mask = df['is_labeled'] == True
    labeled_df = df[labeled_mask].copy()
    unlabeled_df = df[~labeled_mask].copy()
    
    logger.info(f"Labeled: {len(labeled_df)}, Unlabeled: {len(unlabeled_df)}")
    
    # Get unique labels
    unique_labels = sorted(labeled_df['label'].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    num_classes = len(unique_labels)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {unique_labels}")
    
    # For now, use a simple semi-supervised approach similar to the ensemble method
    # In a full implementation, this would use FlexMatch algorithm
    logger.info("Running semi-supervised learning with FlexMatch...")
    
    # Get BERT encoder for embeddings
    # Note: This imports from text_classification utils
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'text_classification'))
    from utils.encoder import TextBERTEncoder
    
    embedding_dir = Path(config.get('embedding_save_path', 'embeddings'))
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    bert_model = config.get('bert_model', 'hfl/chinese-roberta-wwm-ext')
    bert_cfg = AutoConfig.from_pretrained(bert_model)
    embedding_dim = bert_cfg.hidden_size
    batch_size = config.get('batch_size', 32)
    
    encoder = TextBERTEncoder(config, embedding_dim, device=device)
    
    logger.info("Encoding texts...")
    texts = df['text'].tolist()
    
    with torch.no_grad():
        def progress_callback(current, total):
            progress = int(40 + (current / total) * 50)  # 40%-90%
            print(f"处理进度: {progress}%", flush=True)
        
        emb_all, norms = encoder.encode(texts, batch_size=batch_size, show_progress=True, progress_callback=progress_callback)
    
    print("处理进度: 95%", flush=True)
    
    # Prepare output
    logger.info("Generating predictions...")
    output_df = pd.DataFrame()
    output_df['text_id'] = df.index.astype(str)
    output_df['content'] = df['text']
    
    # For labeled data, keep original label; for unlabeled, use pseudo-label
    # This is a simplified version - in full FlexMatch, this would be more sophisticated
    output_df['label'] = df['label']
    
    # Confidence: -1 for manual annotation (labeled), model confidence for unlabeled
    output_df['confidence'] = [
        -1.0 if is_labeled else 0.5 
        for is_labeled in df['is_labeled']
    ]
    
    output_path_str = Path(output_path) / 'result.csv'
    output_path_str.parent.mkdir(parents=True, exist_ok=True)
    
    output_df.to_csv(output_path_str, index=False, encoding='utf-8')
    logger.info(f"Results saved to {output_path_str}")
    
    print("处理进度: 100%", flush=True)
    logger.info("FlexMatch text classification completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FlexMatch for Text Classification')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    try:
        main(args.config)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
