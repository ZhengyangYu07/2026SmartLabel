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
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _resolve_device(device_pref):
    device_name = str(device_pref or 'cpu').strip().lower()
    if device_name.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_name)
        logger.warning('CUDA requested but unavailable in this environment, falling back to CPU.')
        return torch.device('cpu')
    return torch.device(device_name)


class LocalTextBERTEncoder:
    """TorchSSL-main 内部自包含的文本编码器，避免跨目录依赖。"""

    def __init__(self, config, dim, device='cuda'):
        self.device = _resolve_device(device)
        self.out_dim = dim
        self.tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
        self.model = AutoModel.from_pretrained(config['bert_model']).to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=32, show_progress=True, progress_callback=None):
        del show_progress  # 保留接口兼容参数

        all_embeddings = []
        all_norms = []
        total_steps = max((len(texts) + batch_size - 1) // batch_size, 1)

        for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch_texts = texts[start:start + batch_size]
            if not batch_texts:
                continue

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    token_type_ids=encoded.get('token_type_ids')
                )
                pooled = outputs.last_hidden_state[:, 0]

            norm = torch.norm(pooled, p=2, dim=1, keepdim=True)
            normalized = pooled / (norm + 1e-8)

            all_embeddings.append(normalized.cpu())
            all_norms.append(norm.cpu())

            if progress_callback:
                progress_callback(current=batch_idx, total=total_steps)

        if not all_embeddings:
            return torch.empty(0, self.out_dim), torch.empty(0, 1)

        return torch.cat(all_embeddings, dim=0), torch.cat(all_norms, dim=0)


def _compute_uncertainty_stage(label_ratio, config):
    r1 = float(config.get("uncertainty_stage_r1", 0.10))
    r2 = float(config.get("uncertainty_stage_r2", 0.40))
    if label_ratio < r1:
        return "early"
    if label_ratio >= r2:
        return "late"
    return "mid"


def _compute_uncertainty_from_probs(probabilities):
    probs = probabilities.clamp_min(1e-12)
    num_classes = int(probs.size(1))
    top1 = probs.max(dim=1).values

    lowest_conf = (1.0 - top1).clamp(0.0, 1.0)

    if num_classes > 1:
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
    else:
        margin = torch.ones_like(top1)

    bvsb = (1.0 - margin).clamp(0.0, 1.0)

    entropy_raw = -(probs * probs.log()).sum(dim=1)
    entropy_den = torch.log(torch.tensor(float(max(num_classes, 2)), device=probs.device, dtype=probs.dtype))
    entropy = (entropy_raw / entropy_den).clamp(0.0, 1.0)

    return {
        'confidence': top1,
        'lowest_confidence': lowest_conf,
        'margin': margin,
        'bvsb_uncertainty': bvsb,
        'entropy': entropy,
    }


def main(config_path):
    """Main entry point for FlexMatch text classification."""
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    dataset_path = config['dataset_path']
    output_path = config['output_path']
    device = _resolve_device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
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
    
    logger.info("Running semi-supervised learning with prototype-based pseudo probabilities...")
    
    embedding_dir = Path(config.get('embedding_save_path', 'embeddings'))
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    bert_model = config.get('bert_model', 'hfl/chinese-roberta-wwm-ext')
    bert_cfg = AutoConfig.from_pretrained(bert_model)
    embedding_dim = bert_cfg.hidden_size
    batch_size = config.get('batch_size', 32)
    
    encoder = LocalTextBERTEncoder(config, embedding_dim, device=device)
    
    logger.info("Encoding texts...")
    texts = df['text'].tolist()
    
    with torch.no_grad():
        def progress_callback(current, total):
            progress = int(40 + (current / total) * 50)  # 40%-90%
            print(f"处理进度: {progress}%", flush=True)
        
        emb_all, norms = encoder.encode(texts, batch_size=batch_size, show_progress=True, progress_callback=progress_callback)

    emb_all = F.normalize(emb_all, dim=1)
    labeled_indices = torch.tensor(df.index[df['is_labeled'] == True].tolist(), dtype=torch.long)

    if len(labeled_indices) == 0 or num_classes == 0:
        raise ValueError("主题分类至少需要一部分人工标注数据来构建类别原型。")

    label_ids = torch.tensor([label_to_id.get(v, -1) for v in labeled_df['label'].tolist()], dtype=torch.long)
    valid_mask = label_ids >= 0
    if not torch.any(valid_mask):
        raise ValueError("标注数据中的类别无效，无法构建原型。")

    labeled_indices = labeled_indices[valid_mask]
    label_ids = label_ids[valid_mask]

    centroids = []
    for class_id in range(num_classes):
        class_mask = label_ids == class_id
        if torch.any(class_mask):
            class_emb = emb_all[labeled_indices[class_mask]]
            centroid = class_emb.mean(dim=0)
        else:
            centroid = emb_all[labeled_indices].mean(dim=0)
        centroids.append(centroid)

    centroid_tensor = torch.stack(centroids, dim=0)
    centroid_tensor = F.normalize(centroid_tensor, dim=1)

    temperature = float(config.get('prototype_temperature', 0.05))
    temperature = max(temperature, 1e-4)
    logits = emb_all @ centroid_tensor.T
    probs = F.softmax(logits / temperature, dim=1)

    pred_ids = probs.argmax(dim=1).cpu().tolist()
    pred_labels = [id_to_label.get(i, '') for i in pred_ids]

    uncertainty_parts = _compute_uncertainty_from_probs(probs)

    total_count = len(df)
    labeled_count = int(df['is_labeled'].sum())
    label_ratio = (labeled_count / total_count) if total_count > 0 else 0.0
    stage = _compute_uncertainty_stage(label_ratio, config)

    entropy_scores = uncertainty_parts['entropy'].cpu().tolist()
    bvsb_scores = uncertainty_parts['bvsb_uncertainty'].cpu().tolist()
    lowest_conf_scores = uncertainty_parts['lowest_confidence'].cpu().tolist()
    margin_scores = uncertainty_parts['margin'].cpu().tolist()
    confidence_scores = uncertainty_parts['confidence'].cpu().tolist()

    mid_entropy_weight = float(config.get('uncertainty_mid_entropy_weight', 0.7))
    mid_entropy_weight = min(max(mid_entropy_weight, 0.0), 1.0)
    mid_bvsb_weight = 1.0 - mid_entropy_weight

    if stage == 'early':
        stage_uncertainty = entropy_scores
    elif stage == 'mid':
        stage_uncertainty = [
            (mid_entropy_weight * e + mid_bvsb_weight * b)
            for e, b in zip(entropy_scores, bvsb_scores)
        ]
    else:
        stage_uncertainty = lowest_conf_scores

    logger.info(f"Uncertainty stage: {stage}, label_ratio={label_ratio:.3f}")
    
    print("处理进度: 95%", flush=True)
    
    # Prepare output
    logger.info("Generating predictions...")
    output_df = pd.DataFrame()
    output_df['text_id'] = df.index.astype(str)
    output_df['content'] = df['text']
    
    output_labels = []
    output_confidence = []
    output_entropy = []
    output_bvsb = []
    output_margin = []
    output_uncertainty = []
    output_stage = []

    for i, is_labeled in enumerate(df['is_labeled'].tolist()):
        if is_labeled:
            output_labels.append(df.iloc[i]['label'])
            output_confidence.append(-1.0)
            output_entropy.append(-1.0)
            output_bvsb.append(-1.0)
            output_margin.append(-1.0)
            output_uncertainty.append(0.0)
            output_stage.append('labeled')
        else:
            output_labels.append(pred_labels[i])
            output_confidence.append(float(confidence_scores[i]))
            output_entropy.append(float(entropy_scores[i]))
            output_bvsb.append(float(bvsb_scores[i]))
            output_margin.append(float(margin_scores[i]))
            output_uncertainty.append(float(stage_uncertainty[i]))
            output_stage.append(stage)

    output_df['label'] = output_labels
    output_df['confidence'] = output_confidence
    output_df['entropy'] = output_entropy
    output_df['bvsb_uncertainty'] = output_bvsb
    output_df['margin'] = output_margin
    output_df['uncertainty_score'] = output_uncertainty
    output_df['uncertainty_stage'] = output_stage
    
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
