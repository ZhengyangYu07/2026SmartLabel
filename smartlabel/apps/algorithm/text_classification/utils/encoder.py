import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn.functional as F 


DEFAULT_BERT_MODEL_DIR = Path(os.environ.get(
    'SMARTLABEL_TEXT_CLASSIFICATION_BERT_MODEL',
    str(Path(__file__).resolve().parents[5] / 'models' / 'chinese-roberta-wwm-ext'),
))


def _resolve_bert_model_path(model_name_or_path):
    if model_name_or_path and Path(model_name_or_path).exists():
        return model_name_or_path
    if DEFAULT_BERT_MODEL_DIR.exists():
        return str(DEFAULT_BERT_MODEL_DIR)
    return model_name_or_path


def _resolve_device(device_pref):
    device_name = str(device_pref or 'cpu').strip().lower()
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device_name)

class TextBERTEncoder:
    def __init__(self, config, dim, device="cuda",  pooling_type="cls"):
        self.device = _resolve_device(device)
        bert_model_path = _resolve_bert_model_path(config["bert_model"])
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.model = AutoModel.from_pretrained(bert_model_path).to(self.device)
        self.out_dim = dim
        self.pooling_type = pooling_type

    # ==================== 改动部分（为输出当前子进程进度） ====================

    def encode(self, texts, batch_size=32, show_progress=True, progress_callback=None):
        """
        Encodes a list of texts into embeddings.

        Args:
            texts (list of str): The texts to encode.
            batch_size (int): The batch size for encoding.
            show_progress (bool): Whether to display a tqdm progress bar.
            progress_callback (function, optional): A callback function for reporting progress.
                                                   It will be called with (current_step, total_steps).
        """
        all_embeddings = []
        all_norms = []

        # 1. 计算总批次数，用于进度回调
        total_steps = (len(texts) + batch_size - 1) // batch_size
        
        # 2. 创建迭代器，如果 show_progress 为 False，则禁用tqdm
        iterator = tqdm(range(0, len(texts), batch_size), 
                        desc="Encoding Texts", 
                        total=total_steps,
                        disable=not show_progress)
        
        # 使用 enumerate 来获取当前是第几批
        for batch_idx, i in enumerate(iterator):
            batch_texts = texts[i:i + batch_size]
            
            # 检查批次是否为空，以防万一
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
                self.model.eval() # 确保模型在评估模式
                output = self.model(
                    input_ids=encoded_input["input_ids"],
                    attention_mask=encoded_input["attention_mask"],
                    # token_type_ids 对于大多数BERT模型是可选的，这样写更安全
                    token_type_ids=encoded_input.get("token_type_ids")
                )
                hidden = output.last_hidden_state

                # CLS-pooling
                pooled = hidden[:, 0]

            # 计算范数并进行归一化
            norm = torch.norm(pooled, p=2, dim=1, keepdim=True)
            # 防止除以零
            normalized = pooled / (norm + 1e-8)

            all_embeddings.append(normalized.cpu())
            all_norms.append(norm.cpu())
            
            # 3. 在每次循环结束时，调用进度回调函数
            if progress_callback:
                # batch_idx 从 0 开始, 所以传递 batch_idx + 1
                progress_callback(current=batch_idx + 1, total=total_steps)

        # 检查是否有编码结果，避免在空输入时出错
        if not all_embeddings:
            return torch.empty(0, self.out_dim), torch.empty(0, 1)

        return torch.cat(all_embeddings, dim=0), torch.cat(all_norms, dim=0)
