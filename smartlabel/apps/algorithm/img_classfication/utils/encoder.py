import torch
import os
from PIL import Image
from torchvision import transforms
import clip

class ImageCLIPEncoder:
    def __init__(self, config, device):
        self.device = device
        self.image_dir = config["image_dir"]
        self.embedding_save_path = config["embedding_save_path"]
        os.makedirs(self.embedding_save_path, exist_ok=True)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

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
                embeddings.append(features)
        all_features = torch.cat(embeddings, dim=0)
        if save_path:
            torch.save(all_features, os.path.join(save_path, "clip_embeddings.pt"))
        return all_features
