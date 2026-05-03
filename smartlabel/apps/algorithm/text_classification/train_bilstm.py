"""BiLSTM 情感分析训练入口。"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SimpleTextDataset(Dataset):
    def __init__(self, samples):
        # samples: list of (token_ids, label)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def simple_tokenize(text, max_length=128):
    tokens = [token for token in str(text).strip().split() if token]
    if not tokens:
        return [0]
    token_ids = [abs(hash(token)) % 50000 + 1 for token in tokens[:max_length]]
    return token_ids


class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_classes=3, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        pooled = torch.mean(lstm_out, dim=1)
        dropped = self.dropout(pooled)
        return self.fc(dropped)


def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    # eps 避免 log(0)
    eps = 1e-8
    ent = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    return ent


def train_loop(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


def collate_batch(batch):
    texts, labels = zip(*batch)
    max_len = max(len(seq) for seq in texts)
    padded = []
    for seq in texts:
        padded.append(seq + [0] * (max_len - len(seq)))
    return torch.LongTensor(padded), torch.LongTensor(labels)


def load_csv_items(csv_path):
    items = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text') or row.get('content') or ''
            label = row.get('label') or ''
            items.append({'text': text, 'label': label})
    return items


def find_csv_file(input_path):
    path = Path(input_path)
    if path.is_file():
        return path
    csv_files = sorted(path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'未在 {path} 中找到 CSV 文件')
    return csv_files[0]


def build_label_map(label_choices, rows):
    labels = [label for label in label_choices if label]
    if not labels:
        labels = sorted({str(row.get('label') or '').strip() for row in rows if str(row.get('label') or '').strip()})
    if not labels:
        labels = ['正面', '负面', '中性']
    return labels, {label: idx for idx, label in enumerate(labels)}


def make_dataset(rows, label2id, max_length=128):
    samples = []
    for row in rows:
        text = str(row.get('text') or row.get('content') or '')
        label = str(row.get('label') or '').strip()
        if not label:
            continue
        if label not in label2id:
            continue
        token_ids = simple_tokenize(text, max_length=max_length)
        samples.append((token_ids, label2id[label]))
    return samples


def pad_sequences(sequences):
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        padded.append(seq + [0] * (max_len - len(seq)))
    return torch.LongTensor(padded)


def run_training(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    input_path = config.get('text_input_dir') or config.get('dataset_path')
    if not input_path:
        raise ValueError('配置文件中缺少 text_input_dir 或 dataset_path')

    csv_path = find_csv_file(input_path)
    rows = load_csv_items(csv_path)

    label_choices = config.get('label_choices') or []
    labels, label2id = build_label_map(label_choices, rows)
    max_length = int(config.get('max_length', 128))

    labeled_rows = [row for row in rows if str(row.get('label') or '').strip()]
    train_samples = make_dataset(labeled_rows, label2id, max_length=max_length)

    vocab_size = 50001
    embedding_dim = int(config.get('embedding_dim', 128))
    hidden_dim = int(config.get('hidden_dim', 128))
    num_classes = len(labels)
    device = torch.device(config.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = BiLSTMSentiment(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    if train_samples:
        dataloader = DataLoader(train_samples, batch_size=int(config.get('batch_size', 32)), shuffle=True, collate_fn=collate_batch)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get('lr', 1e-3)))
        epochs = int(config.get('epochs', 6))
        for epoch in range(epochs):
            model.train()
            for inputs, target in dataloader:
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                optimizer.step()
            print(f'处理进度: {int((epoch + 1) / epochs * 85)}%', flush=True)

    model.eval()
    result_rows = []
    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            text = str(row.get('text') or row.get('content') or '')
            token_ids = simple_tokenize(text, max_length=max_length)
            input_tensor = pad_sequences([token_ids]).to(device)
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())
            pred_label = labels[pred_idx]
            confidence = float(probs[pred_idx].item())
            result_rows.append({
                'text_id': str(idx),
                'content': text,
                'label': pred_label,
                'confidence': confidence,
            })

    output_path = Path(config.get('output_path') or Path(config_path).parent)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / 'result.csv'
    with open(result_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text_id', 'content', 'label', 'confidence'])
        writer.writeheader()
        writer.writerows(result_rows)

    print('处理进度: 100%', flush=True)
    print(f'结果已保存到: {result_file}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run_training(args.config)
