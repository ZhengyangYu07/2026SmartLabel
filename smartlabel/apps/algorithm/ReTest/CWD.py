# -*- coding: utf-8 -*-
"""@file: CWD.py
@brief: Implementation of Class-Wise Denoising (CWD) for noisy label correction.
@date: 2024-01-01
@version: 1.0   
@details: This module implements the Class-Wise Denoising algorithm to correct noisy labels in a dataset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import Model
from utils.utils import set_device


# def CWD(config, dataloader: DataLoader, dataset, T_hat_init, logger):
#     device = set_device(config["device"])
#     num_classes = dataset.num_classes
#     epochs = config.get("epoch", 30)
#     lr = config.get("lr", 1e-3)
#     weight_decay = config.get("weight_decay", 5e-4)

#     model = Model(config, num_classes=num_classes, MLP_classifier=True, not_ood=True).to(device)
#     optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

#     # 固定 VolMinNet 输出的转移矩阵
#     T_hat = torch.tensor(T_hat_init, dtype=torch.float32, device=device)
#     eps = 1e-8

#     # === Step: 计算固定 class-wise 权重 ===
#     clean_prior = T_hat.sum(dim=1)
#     noisy_prior = T_hat.sum(dim=0)
#     w_c = clean_prior / (noisy_prior + eps)
#     class_weight_tensor = w_c.to(device)

#     logger.debug(f"[CWD] Fixed T_hat diagonal avg: {torch.mean(torch.diag(T_hat)).item() * 100:.2f}%")
#     logger.debug(f"[CWD] Class-wise weights: {w_c.cpu().numpy()}")

#     for epoch in range(epochs):
#         model.train()
#         total_loss, total_samples = 0.0, 0

#         # for x, y, _, _ in tqdm(dataloader, desc=f"[CWD] Epoch {epoch+1}/{epochs}"):
#         for x, y, _, _ in dataloader:
#             x, y = x.to(device), y.to(device)
#             logits = model(x)

#             probs_clean = F.softmax(logits, dim=1)
#             probs_noisy = probs_clean @ T_hat
#             loss_per_sample = F.nll_loss(torch.log(probs_noisy + 1e-10), y, reduction='none')
#             sample_weights = class_weight_tensor[y]
#             loss = (sample_weights * loss_per_sample).mean()

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item() * x.size(0)
#             total_samples += x.size(0)

#         logger.debug(f"[CWD] Epoch {epoch+1} | Loss: {total_loss / total_samples:.4f}")

#     logger.info("CWD training done (using fixed T_hat).")
#     return model

@torch.no_grad()
def _get_feature_dim(backbone, dataloader, device):
    xb, _, _, _ = next(iter(dataloader))
    return backbone(xb.to(device)).shape[1]

def CWD(config, dataloader: DataLoader, dataset, T_hat_init, logger):
    """
    可训练版 CWD（fine-tune backbone）：
      - batch 内用 forward-correction CE:  p_noisy = softmax(logits) @ T_hat
      - epoch 末对分类头 W 执行一次 ld-step:  -2 <W, mu_tilde_S>
      - mu_tilde_S 依论文算法2/式(29)(30)按整轮特征+噪声标签估计（含 Tikhonov 伪逆）

    关键超参（可在 config["CWD_config"] 里覆盖）：
      epoch: int         默认 30
      lr: float          默认 1e-3
      weight_decay: float 默认 5e-4（只用于 W 的 L2）
      cw_pinv_tau: float  默认 1e-3（Tikhonov）
      alpha_ce: float     默认 1.0（CE 权重）
      beta_li: float      默认 0.05（||h||^2 小正则）
      ld_lr_mul: float    默认 10.0（W 的 ld-step 学习率倍率）
      clip_grad: float    默认 1.0
      use_amp: bool       默认 True
    """
    device = set_device(config.get("device", 0))
    C = int(dataset.num_classes)
    epochs = int(config.get("epoch", 30))
    lr = float(config.get("lr", 1e-3))
    wd = float(config.get("weight_decay", 5e-4))
    tau = float(config.get("cw_pinv_tau", 1e-3))
    alpha_ce = float(config.get("alpha_ce", 1.0))
    beta_li  = float(config.get("beta_li", 0.05))
    ld_lr_mul = float(config.get("ld_lr_mul", 10.0))
    clip_grad = float(config.get("clip_grad", 1.0))
    use_amp = bool(config.get("use_amp", True))

    # === Backbone（可训练） ===
    backbone = Model(config, num_classes=C, MLP_classifier=False, not_ood=True).to(device)
    backbone.train()

    # 特征维度 & 线性头
    d = _get_feature_dim(backbone, dataloader, device)
    classifier = nn.Linear(d, C, bias=False).to(device)

    # 优化器
    optim_main = torch.optim.Adam(list(backbone.parameters()) + [classifier.weight], lr=lr, weight_decay=0.0)
    optim_W    = torch.optim.Adam([classifier.weight], lr=lr * ld_lr_mul, weight_decay=0.0)
    # 余弦退火（可注释）
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim_main, T_max=max(epochs, 1))

    # 噪声转移矩阵（行归一化，防止数值漂移）
    eta = torch.tensor(T_hat_init, dtype=torch.float32, device=device)
    eta = torch.clamp(eta, min=0)
    row_sum = eta.sum(dim=1, keepdim=True).clamp(min=1e-12)
    eta = eta / row_sum

    def pinv_tikhonov(M, lam=tau):
        I = torch.eye(M.shape[1], device=M.device, dtype=M.dtype)
        return torch.linalg.solve(M.T @ M + lam * I, M.T)

    @torch.no_grad()
    def estimate_mu_tilde_epoch():
        """按式(29)(30)估计 μ̃(S) ∈ R^{d x C}（整轮常量）"""
        n_total = 0
        mu_hat_tilde = torch.zeros(d, C, device=device)
        count = torch.zeros(C, device=device)

        was_train = backbone.training
        backbone.eval()
        for x, y_noisy, _, _ in dataloader:
            x = x.to(device, non_blocking=True)
            z = backbone(x)  # (B, d)
            y_oh = F.one_hot(y_noisy.to(device), num_classes=C).float()
            mu_hat_tilde += z.T @ y_oh
            count += y_oh.sum(dim=0)
            n_total += x.size(0)
        if was_train: backbone.train()

        n_total = max(1, n_total)
        mu_hat_tilde /= n_total
        p_tilde = count / n_total  # P(\tilde Y)

        # 解先验 π： (eta^T) π = p_tilde
        pi = torch.linalg.lstsq(eta.T, p_tilde).solution
        pi = torch.clamp(pi, min=1e-8); pi = pi / pi.sum()

        I = torch.eye(C, device=device)
        def K_swap(j, k):
            K = I.clone(); K[[j, k]] = K[[k, j]]; return K

        mu_tilde_Sc_list = []
        for c in range(C):
            # π_{c~}
            pi_ctilde = torch.zeros(C, device=device)
            pi_ctilde[c] = pi[c] + (pi * eta[:, c]).sum() - pi[c] * eta[c, c]
            for j in range(C):
                if j == c: continue
                pi_ctilde[j] = (pi * eta[:, j]).sum() - pi[c] * eta[c, j]

            # η'_c
            eta_p = torch.zeros(C, C, device=device)
            for j in range(C):
                if j != c: eta_p[j, j] = 1.0
            denom = torch.clamp(pi[c] + (pi * eta[:, c]).sum() - pi[c] * eta[c, c], min=1e-8)
            for j in range(C):
                if j == c: continue
                eta_p[c, j] = (pi[c] * eta[c, j]) / denom
            eta_p[c, c] = 1.0 - eta_p[c].sum() + eta_p[c, c]

            # M_c
            Mc = torch.zeros(C, C, device=device)
            for j in range(C):
                for k in range(C):
                    val = eta_p[j, k]
                    if val != 0:
                        Mc += pi_ctilde[j] * val * K_swap(j, k).T

            # μ̃(S_c) = μ̂(S̃) @ M_c^†
            M_pinv = pinv_tikhonov(Mc, lam=tau)
            mu_tilde_Sc_list.append(mu_hat_tilde @ M_pinv)

        mu_tilde_S = torch.stack(mu_tilde_Sc_list, dim=0).sum(dim=0) - (C - 1) * mu_hat_tilde
        return mu_tilde_S  # (d x C)

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(epochs):
        # 1) 整轮先估 μ̃(S)
        mu_tilde_S = estimate_mu_tilde_epoch()

        # 2) 主循环：forward-correction CE + 轻量 ||h||^2 + L2(W)
        total_loss, total_samples = 0.0, 0
        for x, y_noisy, _, _ in dataloader:
            x = x.to(device, non_blocking=True)
            y_noisy = y_noisy.to(device, non_blocking=True)

            optim_main.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                z = backbone(x)
                logits = classifier(z)

                # forward-correction CE：给 backbone 提供监督
                p_clean = F.softmax(logits, dim=1)
                p_noisy = torch.clamp(p_clean @ eta, min=1e-12)
                ce = F.nll_loss(torch.log(p_noisy), y_noisy)

                loss = alpha_ce * ce
                if beta_li > 0:
                    li = (logits.pow(2).sum(dim=1)).mean()
                    loss = loss + beta_li * li

                loss = loss + 0.5 * wd * (classifier.weight.pow(2).sum())

            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optim_main)
                torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + [classifier.weight], clip_grad)
            scaler.step(optim_main)
            scaler.update()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        # 3) 轮末：只更新 W 的 ld-step（整轮一次 -2⟨W, μ̃⟩）
        optim_W.zero_grad(set_to_none=True)
        loss_ld = (-2.0) * (classifier.weight.T * mu_tilde_S).sum()
        loss_ld.backward()
        optim_W.step()

        sched.step()
        logger.debug(f"[CWD] Epoch {epoch+1} | Loss(avg): {total_loss / max(1,total_samples):.6f}")

    logger.info("CWD training done (trainable backbone).")

    # === 返回可推理模型 ===
    class CWDModel(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier
        def forward(self, x):
            return self.classifier(self.backbone(x))

    model = CWDModel(backbone, classifier).to(device)
    model.eval()
    return model
