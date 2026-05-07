"""主题分类主动学习的配置约定。"""

from __future__ import annotations


def build_topic_config(
    dataset_path: str,
    output_path: str,
    embedding_save_path: str,
    bert_model: str = "hfl/chinese-roberta-wwm-ext",
    batch_size: int = 32,
    device: str = "cpu",
    min_labeled_per_class: int = 1,
    auto_accept_threshold: float = 0.9,
    uncertainty_top_k: int = 20,
):
    return {
        "dataset_path": dataset_path,
        "dataset_format": "csv",
        "output_path": output_path,
        "embedding_save_path": embedding_save_path,
        "bert_model": bert_model,
        "batch_size": batch_size,
        "device": device,
        "min_labeled_per_class": min_labeled_per_class,
        "auto_accept_threshold": auto_accept_threshold,
        "uncertainty_top_k": uncertainty_top_k,
        "cg3_use": True,
        "cg3_epochs": 200,
        "cg3_lr": 0.001,
        "cg3_k": 15,
        "cg3_threshold": 0.7,
        "cg3_sigma": 0.5,
        "cg3_warmup": 10,
        "llgc_use": False,
        "manifold_use": False,
        "mixtext_use": False,
    }
