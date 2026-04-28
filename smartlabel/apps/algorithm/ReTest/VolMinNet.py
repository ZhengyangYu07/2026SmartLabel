# -*- coding: utf-8 -*-
"""
@file: VolMinNet.py
@brief: Implementation of VolMinNet for estimating transition matrix T from noisy labels.
@date: 2024-01-01
@version: 1.0
@details: This module implements the Volume Minimization Network algorithm to estimate the transition matrix T
            from noisy labels in a dataset. It includes two phases: pretraining a classifier on noisy
            labels and then training the transition matrix with volume regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import Model


class TransitionMatrix(nn.Module):
    def __init__(self, num_classes, temperature: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.logits = nn.Parameter(torch.randn(num_classes, num_classes) * 0.01)
        self._initialize()

    def _initialize(self):
        with torch.no_grad():
            for i in range(self.num_classes):
                self.logits.data[i, i] = 5.0  # 强化对角 dominance

    def forward(self):
        logits = self.logits / self.temperature
        T = torch.softmax(logits, dim=1)  # 每行归一化
        return T


def estimate_transition_matrix(config, dataset, num_classes, logger):
    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = Model(config, num_classes=num_classes, MLP_classifier=False, not_ood=True).to(device)
    T_layer = TransitionMatrix(num_classes, temperature=2.0).to(device)

    lambda_vol = config.get("lambda_vol", 1e-3)
    eps = 1e-6

    # ===== Debug 初始化 T_layer 查看初始转移矩阵 =====
    with torch.no_grad():
        init_T = T_layer().cpu().numpy()
        # logger.debug("初始转移矩阵 T (归一化后):\n%s", np.round(init_T, 4))
        logger.debug("初始对角均值（估计噪声率）: %.2f%%", (1 - np.mean(np.diag(init_T))) * 100)

    # ===== Debug 标签分布 =====
    y_all = [int(y) for _, y, _, _ in dataset]
    y_np = np.array(y_all)
    logger.debug("标签分布:\n%s", np.bincount(y_np, minlength=num_classes))

    # === Phase 1: Pretrain classifier on noisy labels ===
    pretrain_epochs = config.get("pretrain_epoch", 10)
    optimizer1 = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    logger.debug(f"[State] Phase 1: Pretraining classifier for {pretrain_epochs} epochs")
    for epoch in range(pretrain_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        # for x, y, _, _ in tqdm(loader, desc=f"[Pretrain] Epoch {epoch+1}/{pretrain_epochs}"):
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        avg_loss = total_loss / total_samples
        logger.debug(f"[Pretrain] Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    # === Phase 2: Fix classifier, train T with volume regularization ===
    optimizer2 = Adam(T_layer.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    volume_epochs = config.get("volume_epoch", config["epoch"])
    logger.debug(f"[State] Phase 2: Training transition matrix for {volume_epochs} epochs")

    model.eval()
    for epoch in range(volume_epochs):
        T_layer.train()
        total_loss, total_samples = 0, 0

        # for x, y, _, _ in tqdm(loader, desc=f"[Volume] Epoch {epoch+1}/{volume_epochs}"):
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                logits = model(x)
            probs = F.softmax(logits, dim=1)

            T = T_layer()
            noisy_probs = probs @ T
            ce_loss = F.nll_loss(torch.log(noisy_probs + 1e-8), y)
            sign, logdet = torch.slogdet(T + eps * torch.eye(num_classes, device=device))
            if sign <= 0:
                volume_loss = torch.tensor(1e6, device=device)  # 大惩罚
            else:
                volume_loss = -logdet
            loss = ce_loss + lambda_vol * volume_loss

            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        logger.debug(f"[VolMinNet] Epoch {epoch+1} | CE: {ce_loss.item():.4f} | Vol: {volume_loss.item():.4f} | Total: {loss.item():.4f}")

        with torch.no_grad():
            T_hat_epoch = T_layer().detach().cpu().numpy()
            noise_rate_epoch = float(1 - np.mean(np.diag(T_hat_epoch)))
            logger.debug(f"[VolMinNet] Epoch {epoch+1} | Estimated noise rate: {noise_rate_epoch * 100:.2f}%")

    with torch.no_grad():
        T_hat = T_layer().cpu().numpy()
        noise_rate = float(1 - np.mean(np.diag(T_hat)))

        logger.debug("[Final] Learned T matrix:\n%s", np.round(T_hat, 4))
        logger.debug("[Final] Estimated noise rate: %.2f%%", noise_rate * 100)

    return T_hat
