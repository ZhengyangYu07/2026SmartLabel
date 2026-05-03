import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
import sys # 引入 sys 模块用于刷新输出

from utils.data_utils import encode_labels, get_mask
from utils.encoder import TextBERTEncoder
from utils.graph_utils import build_knn_graph, compute_diffusion_map_approx as compute_diffusion_map
from model.CG3_Paper import CG3_Paper
from model.losses import contrastive_loss, structure_generation_loss
from model.LLGC import llgc_predict
from model.ManifoldClassifier import ManifoldClassifier
from model.MixTextClassifier import MixTextClassifier, mixup_data
from transformers import AutoConfig

# ==================== 进度上报辅助函数 ====================
def report_progress(base_progress, current_step, total_steps, stage_span):
    """
    计算并打印当前阶段的总体进度。
    :param base_progress: 当前阶段开始前的基础进度 (e.g., 40 for CG3)
    :param current_step: 当前阶段已完成的步骤 (e.g., epoch)
    :param total_steps: 当前阶段总共的步骤
    :param stage_span: 当前阶段占总进度的百分比 (e.g., 20 for CG3)
    """
    progress = base_progress + (current_step / total_steps) * stage_span
    # 使用 print 并立即刷新，确保Celery能实时捕获
    print(f"处理进度: {int(progress)}%", flush=True)

def denormalize_embeddings(normalized_embeddings, norms):
    return normalized_embeddings * norms.unsqueeze(1)

def run_training(config_path):
    print("开始集成学习...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 自动检测CUDA可用性
    device_str = config.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("警告：CUDA不可用，已自动切换到CPU")
    else:
        device = torch.device(device_str)
    
    batch_size = config.get("batch_size", 32)
    embedding_dir = config.get("embedding_save_path", "embeddings/")
    os.makedirs(embedding_dir, exist_ok=True)
    bert_cfg = AutoConfig.from_pretrained(config["bert_model"])
    embedding_dim = bert_cfg.hidden_size

    # Step 1: Load & encode
    dataset_path = config["dataset_path"]
    print(f"从合并后的数据集文件加载: {dataset_path}")
    
    try:
        all_df = pd.read_csv(dataset_path, encoding='utf-8')
        all_df['label'] = all_df['label'].fillna('') 
        # 处理 is_labeled 列：从 CSV 读出后可能是字符串，需要转换为布尔值
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
    
    # 确保 is_labeled 是布尔值
    # 处理可能的字符串值
    if all_df['is_labeled'].dtype == 'object':
        all_df['is_labeled'] = all_df['is_labeled'].astype(str).str.lower().isin(['true', '1'])
    else:
        all_df['is_labeled'] = all_df['is_labeled'].astype(bool)
        
    labeled_df = all_df[all_df['is_labeled'] == True].copy()
    unlabeled_df = all_df[all_df['is_labeled'] == False].copy()
    print(f"数据加载完成：{len(labeled_df)} 条有标签数据，{len(unlabeled_df)} 条无标签数据。")

    texts_all = all_df["text"].tolist()
    encoder = TextBERTEncoder(config, embedding_dim, device=device)

    print("开始encode...")
    
    # ==================== 2. I/O 优化与进度上报 ====================
    # 不再保存和加载 .pt 文件，直接在内存中处理
    with torch.no_grad():
        # 修改 TextBERTEncoder.encode 方法，让它接受一个回调函数来报告进度
        def progress_callback(current, total):
            report_progress(base_progress=0, current_step=current, total_steps=total, stage_span=40)
        
        # 假设 encoder.encode 支持 progress_callback，如果不支持，需要修改 TextBERTEncoder
        # 这里我们直接使用返回结果
        emb_all, norms = encoder.encode(texts_all, batch_size=batch_size, show_progress=True, progress_callback=progress_callback)

    # 编码完成，进度达到40%
    print("处理进度: 40%")
    print("Encode 完成。")
    X_all = emb_all.to(device) # emb_all 已经是我们需要的张量

    # ==================== 修改结束 ====================

    raw_labels_labeled = labeled_df["label"].astype(str).str.strip().tolist()
    meaningful_labels = all_df[all_df['label'].notna() & (all_df['label'] != '')]['label'].unique().tolist()
    
    labels_labeled, label2id, id2label = encode_labels(
        raw_labels_labeled,
        full_label_list=meaningful_labels
    )
    mask_labeled = get_mask(labels_labeled)
    
    # Step 2: CG3 (进度 40% -> 60%)
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
            
            # ==================== 3. 增加进度上报 ====================
            if epoch % 10 == 0:
                print(f"[CG3] Epoch {epoch:03d}/{cg3_epochs:03d} - Loss: {loss.item():.4f}")
                report_progress(base_progress=40, current_step=epoch, total_steps=cg3_epochs, stage_span=20)
            # ==================== 修改结束 ====================

        print("结束训练CG3...")
        model.eval()
        with torch.no_grad():
            logits = model(X_all, A)["logits"]
            preds = logits.argmax(dim=1).cpu().tolist()
            confidences = torch.softmax(logits, dim=1).max(dim=1).values.cpu().tolist()
        all_df["cg3_pred"] = preds
        all_df["cg3_conf"] = confidences
    
    # Step 3: LLGC (进度 60% -> 65%)
    if config.get("llgc_use", True):
        print("----LLGC-----")
        print("处理进度: 62%") # 中间点
        if not config.get("cg3_use", True):
            print("CG3未启用，LLGC开始独立构图...")
            with torch.no_grad():
                A = build_knn_graph(X_all, k=config["cg3_k"], sigma=config.get("cg3_sigma", 0.5)).coalesce()
                A = torch.sparse_coo_tensor(A.indices(), A.values(), A.size(), device=device)
            print("LLGC构图完成")

        labels_all = [-1] * len(all_df)
        for idx, y in enumerate(labels_labeled): labels_all[idx] = y
        mask_all = torch.tensor([y != -1 for y in labels_all], dtype=torch.bool)
        llgc_preds = llgc_predict(A, labels_all, mask_all, num_classes=len(label2id), alpha=config["llgc_alpha"])
        all_df["llgc_pred"] = torch.argmax(llgc_preds, dim=1).tolist()
        print("LLGC完成")
    report_progress(base_progress=60, current_step=1, total_steps=1, stage_span=5) # 阶段完成

    # Step 4: ManifoldClassifier (进度 65% -> 85%)
    if config.get("manifold_use", True):
        print("-----ManifoldClassifier-----")
        print("计算Diffusion Map...")
        X_all_reduced = compute_diffusion_map(X_all, n_components=config["manifold_components"], sigma=config["manifold_sigma"], n_landmarks=config["manifold_landmarks"]).to(device)
        A_mani = build_knn_graph(X_all_reduced, k=config["manifold_k"], sigma=config["manifold_sigma"]).coalesce()
        A_mani = torch.sparse_coo_tensor(A_mani.indices(), A_mani.values(), A_mani.size(), device=device)
        
        print("标签传播...")
        labels_all = [-1] * len(X_all_reduced)
        for idx, y in enumerate(labels_labeled): labels_all[idx] = y
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
            
            # ==================== 4. 增加进度上报 ====================
            if epoch % 10 == 0:
                print(f"[Manifold] Epoch {epoch:03d} - Loss: {loss.item():.4f}")
                report_progress(base_progress=65, current_step=epoch, total_steps=manifold_epochs, stage_span=20)
            # ==================== 修改结束 ====================

        print("推理阶段...")
        with torch.no_grad():
            model_mani.eval()
            logits = model_mani(X_all_reduced, A_mani)
            all_df["manifold_pred"] = logits.argmax(dim=1).cpu().tolist()
            
    report_progress(base_progress=65, current_step=1, total_steps=1, stage_span=20) # 阶段完成

    # Step 5: MixText (进度 85% -> 95%)
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

            # ==================== 5. 增加进度上报 ====================
            if epoch % 10 == 0:
                print(f"[MixText] Epoch {epoch:03d} - Loss: {loss.item():.4f}")
                report_progress(base_progress=85, current_step=epoch, total_steps=mixtext_epochs, stage_span=10)
            # ==================== 修改结束 ====================

        print("MixText推理中...")
        model_mix.eval()
        with torch.no_grad():
            logits_all = model_mix(X_all)
            all_df["mixtext_pred"] = logits_all.argmax(dim=1).cpu().tolist()
            
    report_progress(base_progress=85, current_step=1, total_steps=1, stage_span=10) # 阶段完成

    # Step 6: 结果生成与保存 (进度 95% -> 100%)
    print("-----保存最终结果-----")
    print("处理进度: 96%")

    if not config.get("cg3_use", True): all_df["cg3_pred"], all_df["cg3_conf"] = -1, 0.0
    if not config.get("llgc_use", True): all_df["llgc_pred"] = -1
    if not config.get("manifold_use", True): all_df["manifold_pred"] = -1
    if not config.get("mixtext_use", True): all_df["mixtext_pred"] = -1

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
    
    # 处理预测标签和标签
    predicted_labels = all_df["predicted_label_index"].apply(lambda x: id2label.get(int(x), "") if pd.notna(x) and x != -1 else "")
    # 有标签数据使用原始 label，无标签数据使用预测 label
    output_df['label'] = [all_df.iloc[i]['label'] if i < len(labeled_df) else predicted_labels.iloc[i] for i in range(len(all_df))]
    
    # 获取置信度
    model_confidence = all_df['cg3_conf'] if 'cg3_conf' in all_df.columns else pd.Series([0.95] * len(all_df), index=all_df.index)
    # 有标签样本置信度设为 -1（表示人工标注），无标签样本使用模型置信度
    output_df['confidence'] = [-1.0 if i < len(labeled_df) else model_confidence.iloc[i] for i in range(len(all_df))]

    output_path = os.path.join(config["output_path"], "result.csv")
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"最终结果（包含有标签和无标签数据）已保存到: {output_path}")

    # Step 7: Cleanup
    # 因为我们不再创建大的 .pt 文件，所以清理步骤可以移除或简化。
    print("-----清理完毕-----")
    print("处理进度: 100%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()
    run_training(args.config)
    print("结束")

