import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
import sys

try:
    from .utils.data_utils import encode_labels, get_mask
    from .utils.encoder import TextBERTEncoder
    from .utils.graph_utils import build_knn_graph, compute_diffusion_map_approx as compute_diffusion_map
    from .models.CG3_Paper import CG3_Paper
    from .models.losses import contrastive_loss, structure_generation_loss
    from .models.LLGC import llgc_predict
    from .models.ManifoldClassifier import ManifoldClassifier
    from .models.MixTextClassifier import MixTextClassifier, mixup_data
except ImportError:
    from utils.data_utils import encode_labels, get_mask
    from utils.encoder import TextBERTEncoder
    from utils.graph_utils import build_knn_graph, compute_diffusion_map_approx as compute_diffusion_map
    from models.CG3_Paper import CG3_Paper
    from models.losses import contrastive_loss, structure_generation_loss
    from models.LLGC import llgc_predict
    from models.ManifoldClassifier import ManifoldClassifier
    from models.MixTextClassifier import MixTextClassifier, mixup_data
from transformers import AutoConfig


def _resolve_device(device_pref):
    device_name = str(device_pref or 'cpu').strip().lower()
    if device_name.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_name)
        print("警告：CUDA不可用，已自动切换到CPU")
        return torch.device('cpu')
    return torch.device(device_name)


def _compute_uncertainty_stage(label_ratio, config):
    r1 = float(config.get("uncertainty_stage_r1", 0.10))
    r2 = float(config.get("uncertainty_stage_r2", 0.40))

    if label_ratio < r1:
        return "early"
    if label_ratio >= r2:
        return "late"
    return "mid"


def _compute_uncertainty_components(prob_tensor):
    probs = prob_tensor.clamp_min(1e-12)
    num_classes = int(probs.size(1))

    top1 = probs.max(dim=1).values
    lowest_confidence = (1.0 - top1).clamp(0.0, 1.0)

    if num_classes > 1:
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
    else:
        margin = torch.ones_like(top1)
    bvsb_uncertainty = (1.0 - margin).clamp(0.0, 1.0)

    entropy_raw = -(probs * probs.log()).sum(dim=1)
    entropy_denom = torch.log(torch.tensor(float(max(num_classes, 2)), device=probs.device, dtype=probs.dtype))
    entropy_uncertainty = (entropy_raw / entropy_denom).clamp(0.0, 1.0)

    return {
        "entropy": entropy_uncertainty,
        "bvsb": bvsb_uncertainty,
        "lowest_confidence": lowest_confidence,
        "margin": margin,
        "top1_confidence": top1,
    }


def report_progress(base_progress, current_step, total_steps, stage_span):
    progress = base_progress + (current_step / total_steps) * stage_span
    print(f"处理进度: {int(progress)}%", flush=True)


def denormalize_embeddings(normalized_embeddings, norms):
    return normalized_embeddings * norms.unsqueeze(1)


def run_training(config_path):
    print("开始集成学习...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = _resolve_device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    batch_size = config.get("batch_size", 32)
    embedding_dir = config.get("embedding_save_path", "embeddings/")
    os.makedirs(embedding_dir, exist_ok=True)
    bert_cfg = AutoConfig.from_pretrained(config["bert_model"])
    embedding_dim = bert_cfg.hidden_size

    dataset_path = config["dataset_path"]
    print(f"从合并后的数据集文件加载: {dataset_path}")

    try:
        all_df = pd.read_csv(dataset_path, encoding='utf-8')
        # 统一标签为字符串，避免数字标签与字符串标签混用导致编码异常
        all_df['label'] = all_df['label'].fillna('').astype(str).str.strip()
        if 'is_labeled' in all_df.columns:
            all_df['is_labeled'] = all_df['is_labeled'].astype(str).str.lower().isin(['true', '1'])
    except FileNotFoundError:
        print(f"错误：数据集文件未找到 at {dataset_path}")
        raise
    except Exception as e:
        print(f"错误：使用pandas读取数据集失败: {e}")
        raise

    if 'is_labeled' not in all_df.columns:
        raise ValueError(f"数据集 {dataset_path} 中缺少 'is_labeled' 列。")

    if all_df['is_labeled'].dtype == 'object':
        all_df['is_labeled'] = all_df['is_labeled'].astype(str).str.lower().isin(['true', '1'])
    else:
        all_df['is_labeled'] = all_df['is_labeled'].astype(bool)

    labeled_df = all_df[all_df['is_labeled'] == True].copy()
    unlabeled_df = all_df[all_df['is_labeled'] == False].copy()
    print(f"数据加载完成：{len(labeled_df)} 条有标签数据，{len(unlabeled_df)} 条无标签数据。")

    total_count = len(all_df)
    labeled_count = len(labeled_df)
    label_ratio = (labeled_count / total_count) if total_count > 0 else 0.0
    uncertainty_stage = _compute_uncertainty_stage(label_ratio, config)
    print(f"不确定性阶段判定: stage={uncertainty_stage}, label_ratio={label_ratio:.3f}")

    # 确保所有文本都是字符串，处理 NaN 和 None
    texts_all = all_df["text"].fillna('').astype(str).tolist()
    encoder = TextBERTEncoder(config, embedding_dim, device=device)

    uncertainty_components = None

    print("开始encode...")
    with torch.no_grad():
        def progress_callback(current, total):
            report_progress(base_progress=0, current_step=current, total_steps=total, stage_span=40)

        emb_all, norms = encoder.encode(texts_all, batch_size=batch_size, show_progress=True, progress_callback=progress_callback)

    print("处理进度: 40%")
    print("Encode 完成。")
    X_all = emb_all.to(device)

    raw_labels_labeled = labeled_df["label"].astype(str).str.strip().tolist()
    meaningful_labels = sorted(set(l for l in all_df['label'].tolist() if l != ''))

    labels_labeled, label2id, id2label = encode_labels(
        raw_labels_labeled,
        full_label_list=meaningful_labels
    )
    mask_labeled = get_mask(labels_labeled)

    if config.get("cg3_use", True):
        print("-----CG3-----")
        print("开始计算图...")
        with torch.no_grad():
            A = build_knn_graph(X_all, k=config["cg3_k"], sigma=config.get("cg3_sigma", 0.5)).coalesce()
            A = torch.sparse_coo_tensor(A.indices(), A.values(), A.size(), device=device)
            assert A.is_sparse, "A is not sparse!"
        print("完成计算图")

        model = CG3_Paper(in_dim=embedding_dim, num_classes=len(label2id)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("cg3_lr", 0.001), weight_decay=config.get("weight_decay", 0))

        labels_tensor = labels_labeled.detach().clone().to(device)
        mask_tensor = torch.tensor(mask_labeled, dtype=torch.bool).detach().clone().to(device)

        print("开始训练CG3...")
        cg3_epochs = config["cg3_epochs"]
        for epoch in range(1, cg3_epochs + 1):
            model.train()
            outputs = model(X_all, A)
            logits = outputs["logits"]
            z1, z2 = outputs["H_local"], outputs["H_global"]
            loss1 = contrastive_loss(z1[:len(labels_labeled)], z2[:len(labels_labeled)], labels_tensor, mask_tensor)
            loss2 = structure_generation_loss(z1, A)
            loss3 = F.cross_entropy(logits[:len(labeled_df)][mask_tensor], labels_tensor[mask_tensor])
            loss = loss1 + loss2 + loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"[CG3] Epoch {epoch:03d}/{cg3_epochs:03d} - Loss: {loss.item():.4f}")
                report_progress(base_progress=40, current_step=epoch, total_steps=cg3_epochs, stage_span=20)

        print("结束训练CG3...")
        model.eval()
        with torch.no_grad():
            logits = model(X_all, A)["logits"]
            preds = logits.argmax(dim=1).cpu().tolist()
            probs = torch.softmax(logits, dim=1)
            confidence_tensor = probs.max(dim=1).values
            uncertainty_components = _compute_uncertainty_components(probs)
            confidences = confidence_tensor.cpu().tolist()
        all_df["cg3_pred"] = preds
        all_df["cg3_conf"] = confidences

    if config.get("llgc_use", True):
        print("----LLGC-----")
        print("处理进度: 62%")
        if not config.get("cg3_use", True):
            print("CG3未启用，LLGC开始独立构图...")
            with torch.no_grad():
                A = build_knn_graph(X_all, k=config["cg3_k"], sigma=config.get("cg3_sigma", 0.5)).coalesce()
                A = torch.sparse_coo_tensor(A.indices(), A.values(), A.size(), device=device)
            print("LLGC构图完成")

        labels_all = [-1] * len(all_df)
        for idx, y in enumerate(labels_labeled):
            labels_all[idx] = y
        mask_all = torch.tensor([y != -1 for y in labels_all], dtype=torch.bool)
        llgc_preds = llgc_predict(A, labels_all, mask_all, num_classes=len(label2id), alpha=config["llgc_alpha"])
        all_df["llgc_pred"] = torch.argmax(llgc_preds, dim=1).tolist()
        print("LLGC完成")
    report_progress(base_progress=60, current_step=1, total_steps=1, stage_span=5)

    if config.get("manifold_use", True):
        print("-----ManifoldClassifier-----")
        print("计算Diffusion Map...")
        X_all_reduced = compute_diffusion_map(X_all, n_components=config["manifold_components"], sigma=config["manifold_sigma"], n_landmarks=config["manifold_landmarks"]).to(device)
        A_mani = build_knn_graph(X_all_reduced, k=config["manifold_k"], sigma=config["manifold_sigma"]).coalesce()
        A_mani = torch.sparse_coo_tensor(A_mani.indices(), A_mani.values(), A_mani.size(), device=device)

        print("标签传播...")
        labels_all = [-1] * len(X_all_reduced)
        for idx, y in enumerate(labels_labeled):
            labels_all[idx] = y
        mask_all = torch.tensor([y != -1 for y in labels_all], dtype=torch.bool, device=device)
        F_soft = llgc_predict(A_mani, labels_all, mask_all, num_classes=len(label2id), alpha=config["llgc_alpha"]).detach()

        print("使用标签传播结果监督训练分类器...")
        model_mani = ManifoldClassifier(in_dim=X_all_reduced.size(1), num_classes=len(label2id), lambda_lap=0.1, k=config["manifold_k"], sigma=config["manifold_sigma"]).to(device)
        optimizer = torch.optim.Adam(model_mani.parameters(), lr=config["manifold_lr"])
        manifold_epochs = config["manifold_epochs"]
        for epoch in range(1, manifold_epochs + 1):
            model_mani.train()
            logits = model_mani(X_all_reduced, A_mani)
            loss = model_mani.get_soft_loss(logits, F_soft, A_mani)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"[Manifold] Epoch {epoch:03d} - Loss: {loss.item():.4f}")
                report_progress(base_progress=65, current_step=epoch, total_steps=manifold_epochs, stage_span=20)

        print("推理阶段...")
        with torch.no_grad():
            model_mani.eval()
            logits = model_mani(X_all_reduced, A_mani)
            all_df["manifold_pred"] = logits.argmax(dim=1).cpu().tolist()

    report_progress(base_progress=65, current_step=1, total_steps=1, stage_span=20)

    if config.get("mixtext_use", True):
        print("-----MixTextClassifier-----")
        X_labeled = X_all[:len(labeled_df)]
        y_labeled = labels_labeled.detach().clone().to(dtype=torch.long, device=device)
        model_mix = MixTextClassifier(in_dim=embedding_dim, num_classes=len(label2id)).to(device)
        optimizer = torch.optim.Adam(model_mix.parameters(), lr=config.get("mixtext_lr", 1e-3))
        mixtext_epochs = config.get("mixtext_epochs", 100)
        for epoch in range(1, mixtext_epochs + 1):
            model_mix.train()
            mixed_x, y_a, y_b, lam = mixup_data(X_labeled, y_labeled, alpha=config.get("mixtext_alpha", 0.4))
            logits = model_mix(mixed_x)
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"[MixText] Epoch {epoch:03d} - Loss: {loss.item():.4f}")
                report_progress(base_progress=85, current_step=epoch, total_steps=mixtext_epochs, stage_span=10)

        print("MixText推理中...")
        model_mix.eval()
        with torch.no_grad():
            logits_all = model_mix(X_all)
            all_df["mixtext_pred"] = logits_all.argmax(dim=1).cpu().tolist()

    report_progress(base_progress=85, current_step=1, total_steps=1, stage_span=10)

    print("-----保存最终结果-----")
    print("处理进度: 96%")

    if not config.get("cg3_use", True):
        all_df["cg3_pred"], all_df["cg3_conf"] = -1, 0.0
    if not config.get("llgc_use", True):
        all_df["llgc_pred"] = -1
    if not config.get("manifold_use", True):
        all_df["manifold_pred"] = -1
    if not config.get("mixtext_use", True):
        all_df["mixtext_pred"] = -1

    final_pred_indices = []
    for index, row in all_df.iterrows():
        if index >= len(labeled_df):
            votes = [row.get("cg3_pred"), row.get("llgc_pred"), row.get("manifold_pred"), row.get("mixtext_pred")]
            valid_votes = [int(v) for v in votes if v is not None and v != -1 and pd.notna(v)]
            final_pred_indices.append(max(set(valid_votes), key=valid_votes.count) if valid_votes else -1)
        else:
            final_pred_indices.append(None)

    all_df["predicted_label_index"] = final_pred_indices

    print("处理进度: 98%")
    output_df = pd.DataFrame()
    output_df['text_id'] = all_df.index.astype(str)
    output_df['content'] = all_df['text']

    predicted_labels = all_df["predicted_label_index"].apply(lambda x: id2label.get(int(x), "") if pd.notna(x) and x != -1 else "")
    output_df['label'] = [all_df.iloc[i]['label'] if i < len(labeled_df) else predicted_labels.iloc[i] for i in range(len(all_df))]

    model_confidence = all_df['cg3_conf'] if 'cg3_conf' in all_df.columns else pd.Series([0.95] * len(all_df), index=all_df.index)

    if uncertainty_components is None:
        confidence_tensor = torch.tensor(model_confidence.astype(float).tolist(), dtype=torch.float32)
        fallback_uncertainty = (1.0 - confidence_tensor).clamp(0.0, 1.0)
        uncertainty_components = {
            "entropy": fallback_uncertainty,
            "bvsb": fallback_uncertainty,
            "lowest_confidence": fallback_uncertainty,
            "margin": (1.0 - fallback_uncertainty).clamp(0.0, 1.0),
            "top1_confidence": confidence_tensor.clamp(0.0, 1.0),
        }

    entropy_scores = uncertainty_components["entropy"].cpu().tolist()
    bvsb_scores = uncertainty_components["bvsb"].cpu().tolist()
    lowest_conf_scores = uncertainty_components["lowest_confidence"].cpu().tolist()
    margin_scores = uncertainty_components["margin"].cpu().tolist()

    mid_entropy_weight = float(config.get("uncertainty_mid_entropy_weight", 0.7))
    mid_entropy_weight = min(max(mid_entropy_weight, 0.0), 1.0)
    mid_bvsb_weight = 1.0 - mid_entropy_weight

    if uncertainty_stage == "early":
        stage_uncertainty_scores = entropy_scores
    elif uncertainty_stage == "mid":
        stage_uncertainty_scores = [
            (mid_entropy_weight * e + mid_bvsb_weight * b)
            for e, b in zip(entropy_scores, bvsb_scores)
        ]
    else:
        stage_uncertainty_scores = lowest_conf_scores

    output_df['confidence'] = [-1.0 if i < len(labeled_df) else model_confidence.iloc[i] for i in range(len(all_df))]
    output_df['entropy'] = [-1.0 if i < len(labeled_df) else entropy_scores[i] for i in range(len(all_df))]
    output_df['bvsb_uncertainty'] = [-1.0 if i < len(labeled_df) else bvsb_scores[i] for i in range(len(all_df))]
    output_df['margin'] = [-1.0 if i < len(labeled_df) else margin_scores[i] for i in range(len(all_df))]
    output_df['uncertainty_score'] = [0.0 if i < len(labeled_df) else stage_uncertainty_scores[i] for i in range(len(all_df))]
    output_df['uncertainty_stage'] = ['labeled' if i < len(labeled_df) else uncertainty_stage for i in range(len(all_df))]

    output_path = os.path.join(config["output_path"], "result.csv")
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"最终结果（包含有标签和无标签数据）已保存到: {output_path}")
    print("-----清理完毕-----")
    print("处理进度: 100%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()
    run_training(args.config)
    print("结束")
