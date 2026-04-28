

import os
import json
import math
import torch
import torch.nn.functional as F
# from torch.utils.data import DataLoader

from utils.data_utils import load_image_dataset, encode_labels, get_mask
from utils.encoder import ImageCLIPEncoder
from utils.graph_utils import build_knn_graph, compute_diffusion_map

from model.CG3_Paper import CG3_Paper
from model.DiffMAP import DiffusionMapModel
from model.LLGC import llgc_predict
from model.ManifoldClassifier import ManifoldClassifier, ManifoldClassifierWithDropout


def emit_progress(progress):
    progress = max(0, min(int(progress), 99))
    print(f"处理进度: {progress}%")


def train(config):
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    emit_progress(22)

    # 加载数据
    labeled_df, unlabeled_df = load_image_dataset(image_dir=config["image_dir"] ,label_csv=config["label_csv"],
    split=True)
    image_names = list(labeled_df["filepath"]) + list(unlabeled_df["filepath"])
    labels = list(labeled_df["label"]) + [-1] * len(unlabeled_df)  # 未标注样本标签-1
    print(f"[Debug] image_names = {image_names[:5]}, total = {len(image_names)}")
    emit_progress(26)

    encoded_labels, label2id, id2label = encode_labels(labels)
    mask = get_mask(encoded_labels).to(device)  # mask=True 表示有标签样本

    # 生成图像特征向量
    encoder = ImageCLIPEncoder(config, device)
    features = encoder.encode(image_names, batch_size=config["batch_size"], save_path=config["embedding_save_path"])
    features = features.float()  # 强制 float32，避免后续图操作报错
    features = features.to(device)
    emit_progress(35)
    
    N = features.size(0)
    num_classes = len(label2id)
    print(f"[Info] Total samples: {N}, Classes: {num_classes}")
    print(f"[Info] Labeled samples: {mask.sum().item()}, Unlabeled samples: {(~mask).sum().item()}")

    # 构建图结构
    W_knn = build_knn_graph(features, k=config["cg3_k"]).to(device)
    W_diffmap = compute_diffusion_map(features, sigma=config["diffmap_sigma"], k=config["diffmap_k"]).to(device)
    emit_progress(40)

    # 转换标签张量
    labels_tensor = torch.tensor(encoded_labels, dtype=torch.long, device=device)

    ###############################################
    # CG3_Paper 训练与推理
    ###############################################
    cg3_model = CG3_Paper(in_dim=features.size(1),
                          hidden_dim=config["cg3_hidden_dim"],
                          out_dim=config["cg3_out_dim"],
                          num_classes=num_classes, num_layers=config["cg3_num_layers"]).to(device)
    optimizer_cg3 = torch.optim.Adam(
        cg3_model.parameters(), lr=config["cg3_lr"])
    for epoch in range(config["cg3_epochs"]):
        cg3_model.train()
        optimizer_cg3.zero_grad()
        logits = cg3_model(features, W_knn, W_diffmap)["logits"]
        loss = F.cross_entropy(logits[mask], labels_tensor[mask])
        loss.backward()
        optimizer_cg3.step()
        if (epoch + 1) % (config["cg3_epochs"]/10) == 0 or epoch == 0:
            print(f"[CG3] Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            cg3_progress = 40 + int((epoch + 1) / max(config["cg3_epochs"], 1) * 30)
            emit_progress(cg3_progress)
    cg3_model.eval()
    with torch.no_grad():
        logits_cg3 = cg3_model(features, W_knn, W_diffmap)["logits"]
        preds_cg3 = logits_cg3.argmax(dim=1)

    ###############################################
    # LLGC 推理
    ###############################################
    preds_llgc_score = llgc_predict(
        W_knn, labels_tensor, mask, num_classes=num_classes, alpha=config["llgc_alpha"])
    preds_llgc = preds_llgc_score.argmax(dim=1)
    emit_progress(72)

    ###############################################
    # Manifold Regularization 训练与推理
    ###############################################
    manifold_input = F.softmax(logits_cg3, dim=1)
    manifold_model = ManifoldClassifier(in_dim=num_classes, num_classes=num_classes,
                                        hidden_dims=config["manifold_hidden_dims"], lambda_lap=config["manifold_lambda"]).to(device)
    optimizer_manifold = torch.optim.Adam(
        manifold_model.parameters(), lr=config["manifold_lr"])

    for epoch in range(config["manifold_epochs"]):
        manifold_model.train()
        optimizer_manifold.zero_grad()
        logits = manifold_model(manifold_input, W_knn)
        loss = manifold_model.get_loss(logits, labels_tensor, mask, W_knn)
        loss.backward()
        optimizer_manifold.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = ((preds == labels_tensor) & mask).sum().item()
            total = mask.sum().item()
            acc = correct / total if total > 0 else 0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[Manifold] Epoch {epoch + 1}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}")
            manifold_progress = 72 + int((epoch + 1) / max(config["manifold_epochs"], 1) * 20)
            emit_progress(manifold_progress)
    manifold_model.eval()
    with torch.no_grad():
        logits_manifold = manifold_model(manifold_input, W_knn)
        preds_manifold = logits_manifold.argmax(dim=1)

    ###############################################
    # 多数投票融合
    ###############################################
    preds_final = []
    for p1, p2, p3 in zip(preds_cg3, preds_llgc, preds_manifold):
        votes = [p1.item(), p2.item(), p3.item()]
        final = max(set(votes), key=votes.count)
        preds_final.append(final)
    preds_final = torch.tensor(preds_final, device=device)
    emit_progress(93)


    #######################################################
    #输出结果
    #######################################################
    # 输出预测结果包括所有标注
    output_dir = config["output_path"]
    os.makedirs(output_dir, exist_ok=True)
    image_names_only = [os.path.basename(p) for p in image_names]
    image_paths = image_names
    probs_final = F.softmax(logits_manifold, dim=1)

    records = []
    for img_name, img_path, m, p1, p2, p3, pf, prob in zip(
            image_names_only, image_paths, mask, preds_cg3, preds_llgc, preds_manifold, preds_final, probs_final):
        # m (mask) 为 True 表示有标签样本（已标注样本）
        # m (mask) 为 False 表示无标签样本

        label1 = id2label.get(p1.item(), "unknown")
        label2 = id2label.get(p2.item(), "unknown")
        label3 = id2label.get(p3.item(), "unknown")
        labelf = id2label.get(pf.item(), "unknown")

        # 根据 mask 判断是否为已标注样本，并设置置信度
        if m: # 如果是已标注样本
            conf = -1.0 # 置信度设置为 -1
            entropy = -1.0
            top1_prob = -1.0
            top2_prob = -1.0
            margin = -1.0
            uncertainty_score = -1.0
        else: # 如果是无标签样本
            conf = prob[pf].item() # 正常计算置信度
            prob_values = prob.detach().float()
            topk = torch.topk(prob_values, k=min(2, prob_values.numel()))
            top1_prob = float(topk.values[0].item()) if topk.values.numel() > 0 else 0.0
            top2_prob = float(topk.values[1].item()) if topk.values.numel() > 1 else 0.0
            entropy_raw = float(-(prob_values * torch.log(prob_values.clamp_min(1e-12))).sum().item())
            entropy_denominator = math.log(max(int(prob_values.numel()), 2))
            entropy = entropy_raw / entropy_denominator if entropy_denominator > 0 else 0.0
            margin = top1_prob - top2_prob
            uncertainty_score = entropy

        records.append((img_name, img_path, label1, label2,
                       label3, labelf, f"{conf:.4f}", f"{entropy:.6f}", f"{margin:.6f}", f"{top1_prob:.6f}", f"{top2_prob:.6f}", f"{uncertainty_score:.6f}")) # 增加熵、边际和不确定性分数

    records.sort(key=lambda x: x[0])
    with open(os.path.join(output_dir, "result.csv"), "w", encoding="utf-8") as f:
        f.write(
            "image_name,image_path,pred_cg3,pred_llgc,pred_manifold,label,confidence,entropy,margin,top1_prob,top2_prob,uncertainty_score\n")
        for rec in records:
            f.write(",".join(map(str, rec)) + "\n") # 确保所有元素都被转换为字符串
    emit_progress(98)
    print(
        f"[Done] Prediction saved to {os.path.join(output_dir, 'result.csv')}")
    # ###############################################
    # # 输出准确率
    # ###############################################
    # import csv
    # true_label_file = "/home/smallb/models/img_classfication1/data/cifar10_images_batch1_10000/labels.csv"
    # true_labels = {}
    # with open(true_label_file, "r", encoding="utf-8") as f:
    #     reader = csv.reader(f)
    #     next(reader)
    #     for row in reader:
    #         fname, label = row
    #         true_labels[fname] = label

    # correct_cg3 = correct_llgc = correct_manifold = correct_final = total = 0
    # for img_name, m, p1, p2, p3, pf in zip(image_names_only, mask, preds_cg3, preds_llgc, preds_manifold, preds_final):
    #     if not m:
    #         total += 1
    #         true = true_labels.get(img_name, None)
    #         if true is None:
    #             continue
    #         if id2label[p1.item()] == true:
    #             correct_cg3 += 1
    #         if id2label[p2.item()] == true:
    #             correct_llgc += 1
    #         if id2label[p3.item()] == true:
    #             correct_manifold += 1
    #         if id2label[pf.item()] == true:
    #             correct_final += 1

    # print(f"Accuracy CG3: {correct_cg3 / total:.4f}")
    # print(f"Accuracy LLGC: {correct_llgc / total:.4f}")
    # print(f"Accuracy Manifold: {correct_manifold / total:.4f}")
    # print(f"Accuracy Final Fusion (Voting): {correct_final / total:.4f}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/smallb/models/img_classfication/configs/config_img.json", help="Path to config JSON file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    train(config)
