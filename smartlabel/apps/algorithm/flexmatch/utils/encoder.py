"""BERT 文本编码器。

把原始文本批量编码为 L2 归一化的句向量（[CLS] 表示）并返回原始模长，
是整条半监督分类流程的输入表示层。
"""
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def _resolve_device(device_pref):
    """解析设备偏好，指定 cuda 但不可用时回退 CPU。"""
    device_name = str(device_pref or 'cpu').strip().lower()
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device_name)


class TextBERTEncoder:
    def __init__(self, config, dim, device="cuda", pooling_type="cls"):
        """加载分词器与 BERT 模型，记录输出维度与池化方式。"""
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])
        self.model = AutoModel.from_pretrained(config["bert_model"]).to(self.device)
        self.out_dim = dim
        self.pooling_type = pooling_type

    def encode(self, texts, batch_size=32, show_progress=True, progress_callback=None):
        """分批前向编码文本，返回归一化句向量与对应的原始模长。"""
        all_embeddings = []
        all_norms = []
        total_steps = (len(texts) + batch_size - 1) // batch_size

        iterator = tqdm(range(0, len(texts), batch_size),
                        desc="Encoding Texts",
                        total=total_steps,
                        disable=not show_progress)

        for batch_idx, i in enumerate(iterator):
            batch_texts = texts[i:i + batch_size]
            if not batch_texts:
                continue

            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                self.model.eval()
                output = self.model(
                    input_ids=encoded_input["input_ids"],
                    attention_mask=encoded_input["attention_mask"],
                    token_type_ids=encoded_input.get("token_type_ids")
                )
                hidden = output.last_hidden_state
                # 取 [CLS] 位置的隐状态作为整句表示
                pooled = hidden[:, 0]

            # L2 归一化，使余弦相似度等价于点积，便于后续构图与对比损失
            norm = torch.norm(pooled, p=2, dim=1, keepdim=True)
            normalized = pooled / (norm + 1e-8)

            all_embeddings.append(normalized.cpu())
            # 保留原始模长，供 train.py 反归一化还原
            all_norms.append(norm.cpu())

            if progress_callback:
                progress_callback(current=batch_idx + 1, total=total_steps)

        if not all_embeddings:
            return torch.empty(0, self.out_dim), torch.empty(0, 1)

        return torch.cat(all_embeddings, dim=0), torch.cat(all_norms, dim=0)
