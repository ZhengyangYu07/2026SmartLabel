import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
import clip


class ImageCLIPEncoder:
    def __init__(self, config, device):
        self.device = device
        self.config = config or {}
        self.image_dir = config["image_dir"]
        self.embedding_save_path = config["embedding_save_path"]
        os.makedirs(self.embedding_save_path, exist_ok=True)

        clip_model_name = self.config.get("clip_model", "ViT-B/32")
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)

        # adapter 配置可以通过 config['adapter'] 传入，示例：
        # {"enabled": True, "num_layers": 1, "hidden_dim": 512, "activation": "relu"}
        self.adapter_config = self.config.get("adapter", {}) or {}
        self.adapter = None  # 延迟构建，因为需要知道 CLIP 输出维度

    def _build_adapter(self, in_dim: int, cfg: dict) -> nn.Module:
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            return None

        num_layers = int(cfg.get("num_layers", 1))
        hidden_dim = int(cfg.get("hidden_dim", max(256, in_dim)))
        activation = cfg.get("activation", "relu").lower()

        if activation == "gelu":
            act = nn.GELU()
        else:
            act = nn.ReLU()

        layers = []
        if num_layers <= 0:
            return None

        # 构造： in_dim -> hidden_dim -> ... -> hidden_dim -> in_dim
        # 最后一层映射回 in_dim，保持特征维度不变
        # 第一层
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act)

        # 中间层（如果 num_layers > 1）
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)

        # 最后一投影回原始维度
        layers.append(nn.Linear(hidden_dim, in_dim))

        return nn.Sequential(*layers)

    def encode(self, image_names, batch_size=32, save_path=None):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(image_names), batch_size):
                batch_paths = image_names[i:i+batch_size]
                images = []
                for name in batch_paths:
                    path = os.path.join(self.image_dir, name)
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(self.preprocess(img))
                    except Exception as e:
                        print(f"[Warning] Failed to load image: {path} ({e})")
                        images.append(torch.zeros(3, 224, 224))
                images = torch.stack(images).to(self.device)
                features = self.model.encode_image(images)

                # 强制为 float 并移到 device
                features = features.float().to(self.device)

                # 延迟构建 adapter（需要知道 features 的维度）
                if self.adapter is None and self.adapter_config.get("enabled", False):
                    in_dim = features.shape[1]
                    adapter = self._build_adapter(in_dim, self.adapter_config)
                    if adapter is not None:
                        self.adapter = adapter.to(self.device)

                # 应用 adapter
                if self.adapter is not None:
                    features = self.adapter(features)

                embeddings.append(features)
        all_features = torch.cat(embeddings, dim=0)
        if save_path:
            torch.save(all_features, os.path.join(save_path, "clip_embeddings.pt"))
        return all_features
