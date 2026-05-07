import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def _resolve_device(device_pref):
    device_name = str(device_pref or 'cpu').strip().lower()
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device_name)


class TextBERTEncoder:
    def __init__(self, config, dim, device="cuda", pooling_type="cls"):
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])
        self.model = AutoModel.from_pretrained(config["bert_model"]).to(self.device)
        self.out_dim = dim
        self.pooling_type = pooling_type

    def encode(self, texts, batch_size=32, show_progress=True, progress_callback=None):
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
                pooled = hidden[:, 0]

            norm = torch.norm(pooled, p=2, dim=1, keepdim=True)
            normalized = pooled / (norm + 1e-8)

            all_embeddings.append(normalized.cpu())
            all_norms.append(norm.cpu())

            if progress_callback:
                progress_callback(current=batch_idx + 1, total=total_steps)

        if not all_embeddings:
            return torch.empty(0, self.out_dim), torch.empty(0, 1)

        return torch.cat(all_embeddings, dim=0), torch.cat(all_norms, dim=0)
