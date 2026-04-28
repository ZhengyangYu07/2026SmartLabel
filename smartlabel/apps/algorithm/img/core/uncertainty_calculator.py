# core/uncertainty_calculator.py
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
import numpy as np


class UncertaintyCalculator:
    """
    不确定性计算器 - 支持多种不确定性度量方法
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def calculate_uncertainty(self, model, dataloader, method="entropy"):
        """
        计算未标注数据的不确定性

        Args:
            model: 训练好的模型
            dataloader: 未标注数据的DataLoader
            method: 不确定性计算方法 - "entropy", "least_confidence", "margin"

        Returns:
            List[float]: 每个样本的不确定性分数
        """
        model.eval()
        uncertainties = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]  # (images, labels) 的情况
                else:
                    inputs = batch  # 只有images的情况

                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)

                if method == "entropy":
                    batch_uncertainties = self._entropy_uncertainty(probabilities)
                elif method == "least_confidence":
                    batch_uncertainties = self._least_confidence_uncertainty(probabilities)
                elif method == "margin":
                    batch_uncertainties = self._margin_uncertainty(probabilities)
                else:
                    raise ValueError(f"未知的不确定性计算方法: {method}")

                uncertainties.extend(batch_uncertainties.cpu().numpy())

        return uncertainties

    def _entropy_uncertainty(self, probabilities):
        """基于信息熵的不确定性计算"""
        return -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)

    def _least_confidence_uncertainty(self, probabilities):
        """基于最低置信度的不确定性计算"""
        max_probs, _ = torch.max(probabilities, dim=1)
        return 1 - max_probs

    def _margin_uncertainty(self, probabilities):
        """基于间隔的不确定性计算"""
        top2_probs, _ = torch.topk(probabilities, 2, dim=1)
        return 1 - (top2_probs[:, 0] - top2_probs[:, 1])

    def select_most_uncertain(self, uncertainties, top_k=10):
        """
        选择最不确定的样本

        Args:
            uncertainties: 不确定性分数列表
            top_k: 选择前K个最不确定的样本

        Returns:
            List[int]: 选中的样本索引
        """
        indices = np.argsort(uncertainties)[::-1]  # 降序排列
        return indices[:top_k].tolist()

    def select_most_confident(self, uncertainties, threshold=0.95):
        """
        选择最置信的样本（用于伪标签）

        Args:
            uncertainties: 不确定性分数列表（这里使用预测概率）
            threshold: 置信度阈值

        Returns:
            List[int]: 选中的样本索引
        """
        # 这里uncertainties应该是预测概率
        confident_indices = []
        for idx, prob in enumerate(uncertainties):
            if prob >= threshold:
                confident_indices.append(idx)
        return confident_indices