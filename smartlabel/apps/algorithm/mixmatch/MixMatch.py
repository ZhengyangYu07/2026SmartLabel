import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import itertools

# === 1. 解析命令行参数 ===
def parse_args():
    parser = argparse.ArgumentParser(description="MixMatch Training")
    parser.add_argument("--json", type=str, default="default.json", help="Path to the JSON configuration file")
    return parser.parse_args()

# === 2. 读取 JSON 配置文件 ===
def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

# === 3. 自定义未标注数据集 ===
class UnlabeledDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# === 4. 伪标签处理（温度缩放） ===
def sharpen(probs, T=0.5):
    probs = probs ** (1 / T)
    return probs / probs.sum(dim=1, keepdim=True)

# === 5. MixUp 数据增强 ===
def mixup(x1, y1, x2, y2, alpha=0.5):
    lam = np.random.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2

# === 6. 生成伪标签 ===
def generate_pseudo_labels(model, unlabeled_loader, num_classes, device, K=2, T=0.5):
    model.eval()
    pseudo_labels, images_list = [], []
    with torch.no_grad():
        for images in unlabeled_loader:
            images = images.to(device)
            preds = torch.stack([torch.softmax(model(images), dim=1) for _ in range(K)], dim=0).mean(dim=0)
            sharpened_preds = sharpen(preds, T)
            one_hot_labels = torch.zeros_like(sharpened_preds)
            one_hot_labels.scatter_(1, sharpened_preds.argmax(dim=1, keepdim=True), 1)
            pseudo_labels.append(one_hot_labels)
            images_list.append(images)
    return torch.cat(images_list), torch.cat(pseudo_labels)

# === 7. 简单CNN 分类模型 ===
class DeepCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === 8. 获取数据加载器 ===
def get_dataloaders(dataset_path, batch_size=64):
    # 计算数据集的均值和标准差
    def calculate_mean_and_std(dataset_path, batch_size=64):
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        mean = 0.
        std = 0.
        total_images_count = 0

        for images, _ in dataloader:
            batch_samples = images.size(0)  # 当前 batch 的样本数
            images = images.view(batch_samples, 3, -1)  # 展平每个图像
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images_count += batch_samples

        mean /= total_images_count
        std /= total_images_count

        return mean.tolist(), std.tolist()

    # 获取均值和标准差
    mean, std = calculate_mean_and_std(dataset_path)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    labeled_dataset = datasets.ImageFolder(os.path.join(dataset_path, "labeled"), transform=transform)
    unlabeled_dataset = UnlabeledDataset(os.path.join(dataset_path, "unlabeled"), transform=transform)

    batch_size = min(len(labeled_dataset), len(unlabeled_dataset), batch_size)  # 取最小 batch_size 以匹配

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return labeled_loader, unlabeled_loader, labeled_dataset.classes

# === 9. 训练 MixMatch ===
def train_mixmatch(model, labeled_loader, unlabeled_loader, num_classes, epochs=400, device="cuda"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 正则化
    criterion = nn.CrossEntropyLoss()

    labeled_iter = iter(labeled_loader)  # 创建 labeled_loader 迭代器
    warmup_epochs = int(0.2 * epochs)

    # 1. 进行 warm-up 阶段
    for epoch in range(warmup_epochs):
        model.train()
        total_loss = 0
        for labeled_images, labels in labeled_loader:
            labeled_images, labels = labeled_images.to(device), labels.to(device)

            outputs = model(labeled_images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Warm-up Epoch [{epoch+1}/{warmup_epochs}], Loss: {total_loss:.4f}")

    # 2. 进入正式训练阶段，使用 MixMatch
    for epoch in range(warmup_epochs, epochs):
        model.train()
        total_loss = 0

        unlabeled_images, pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, num_classes, device)

        for unlabeled_images_batch, pseudo_labels_batch in zip(unlabeled_images.split(64), pseudo_labels.split(64)):
            try:
                labeled_images, labels = next(labeled_iter)  # 获取新的 labeled batch
            except StopIteration:
                labeled_iter = iter(labeled_loader)  # 重新创建迭代器
                labeled_images, labels = next(labeled_iter)

            labeled_images, labels = labeled_images.to(device), labels.to(device)
            unlabeled_images_batch, pseudo_labels_batch = unlabeled_images_batch.to(device), pseudo_labels_batch.to(device)

            mixed_images, mixed_labels = mixup(
                labeled_images, torch.eye(num_classes, device=device)[labels],
                unlabeled_images_batch, pseudo_labels_batch
            )

            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    return model

# === 10. 预测所有图片并保存 CSV ===
def predict_all_images(model, dataset_path, class_names, output_csv="predictions.csv", device="cuda"):
    model.eval()
    
    # 动态计算数据集的均值和标准差
    def calculate_mean_and_std(dataset_path, batch_size=64):
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        mean = 0.
        std = 0.
        total_images_count = 0

        for images, _ in dataloader:
            batch_samples = images.size(0)  # 当前 batch 的样本数
            images = images.view(batch_samples, 3, -1)  # 展平每个图像
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images_count += batch_samples

        mean /= total_images_count
        std /= total_images_count

        return mean.tolist(), std.tolist()

    # 获取均值和标准差
    mean, std = calculate_mean_and_std(dataset_path)
    
    # 设置变换
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    results = []
    for subdir in ["labeled", "unlabeled"]:
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.exists(subdir_path):  
            continue  

        for root, _, files in os.walk(subdir_path):  # 递归遍历
            for img_name in files:
                img_path = os.path.join(root, img_name)
                if not img_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):  # 只处理图片
                    continue

                try:
                    image = Image.open(img_path).convert("RGB")
                    image = transform(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        pred = model(image).argmax(dim=1).item()

                    results.append([img_name, class_names[pred]])
                except Exception as e:
                    print(f"读取 {img_path} 失败: {e}")

    pd.DataFrame(results, columns=["image_name", "predicted_label"]).to_csv(output_csv, index=False)
    print(f"预测结果已保存到 {output_csv}")


# === 11. 主程序 ===
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.json)  # 解析 --json 传入的 JSON 配置

    dataset_path = config["dataset_path"]
    output_csv = config["output_csv"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled_loader, unlabeled_loader, class_names = get_dataloaders(dataset_path)
    num_classes = len(class_names)

    model = DeepCNN(num_classes)
    trained_model = train_mixmatch(model, labeled_loader, unlabeled_loader, num_classes, device=device)
    predict_all_images(trained_model, dataset_path, class_names, output_csv=output_csv)
