"""主动学习采样与自动接收策略。"""

from __future__ import annotations

from typing import Iterable, List


def pick_high_uncertainty(indices: Iterable[int], scores: Iterable[float], top_k: int = 20) -> List[int]:
    ranked = sorted(zip(indices, scores), key=lambda item: item[1], reverse=True)
    return [index for index, _ in ranked[:top_k]]


def should_auto_accept(confidence: float, threshold: float, class_covered: bool) -> bool:
    return class_covered and confidence >= threshold

