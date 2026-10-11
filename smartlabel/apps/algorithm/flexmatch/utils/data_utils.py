"""数据集加载与标签编码工具。

把 csv/json/txt 原始数据读成 DataFrame、拆分有/无标签样本，并把字符串标签映射为整数 id，
是 train.py 半监督文本分类流程的数据输入层。
"""
import pandas as pd
import torch
import os


def load_dataset(path, file_format="csv", split=True):
    """从单个文件读取数据；split=True 时按标签是否为空拆成有标签/无标签两份。"""
    if file_format == "csv":
        df = pd.read_csv(path)

    elif file_format == "json":
        df = pd.read_json(path, lines=True)

    elif file_format == "txt":
        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('"') and "," in line:
                    try:
                        text_part, label = line.rsplit(",", 1)
                        text = text_part.strip('"')
                        labels.append(label.strip())
                        texts.append(text.strip())
                    except ValueError:
                        continue
        df = pd.DataFrame({"text": texts, "label": labels})

    else:
        raise ValueError("Unsupported file format")

    df["text"] = df["text"].astype(str)
    # 统一标签为字符串，空串和 "nan" 都视为无标签
    df["label"] = df["label"].astype(str).str.strip().replace("nan", "")

    if split:
        # 标签非空为有标签样本，空为待预测的无标签样本
        labeled_df = df[df["label"] != ""].reset_index(drop=True)
        unlabeled_df = df[df["label"] == ""].reset_index(drop=True)
        return labeled_df, unlabeled_df
    else:
        return df


def encode_labels(labels, full_label_list=None):
    """把字符串标签映射为整数 id，未在词表中的标签编码为 -1（表示无标签）。"""
    if full_label_list is None:
        label_set = sorted(set(l for l in labels if l != ""))
    else:
        # 用全量标签词表，保证有/无标签样本共享同一套 id 空间
        label_set = sorted(set(l for l in full_label_list if l != ""))

    label2id = {l: i for i, l in enumerate(label_set)}
    id2label = {i: l for l, i in label2id.items()}
    # -1 作为无标签占位符，后续用 get_mask 区分
    encoded = torch.LongTensor([label2id[l] if l in label2id else -1 for l in labels])
    return encoded, label2id, id2label


def get_mask(encoded_labels):
    """生成布尔掩码，True 表示该样本有真实标签（可参与监督损失）。"""
    return encoded_labels != -1


def load_dataset_from_dir(folder_path, file_format="csv", split=True):
    """读取目录下所有同格式文件并合并，再按标签是否为空拆分。"""
    all_dfs = []

    for fname in os.listdir(folder_path):
        if not fname.endswith(f".{file_format}"):
            continue

        full_path = os.path.join(folder_path, fname)

        if file_format == "csv":
            df = pd.read_csv(full_path)
        elif file_format == "json":
            df = pd.read_json(full_path, lines=True)
        elif file_format == "txt":
            texts, labels = [], []
            with open(full_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('"') and "," in line:
                        try:
                            text_part, label = line.rsplit(",", 1)
                            text = text_part.strip('"')
                            labels.append(label.strip())
                            texts.append(text.strip())
                        except ValueError:
                            continue
            df = pd.DataFrame({"text": texts, "label": labels})
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(str).str.strip().replace("nan", "")
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No valid .{file_format} files found in {folder_path}")

    full_df = pd.concat(all_dfs, ignore_index=True)

    if split:
        labeled_df = full_df[full_df["label"] != ""].reset_index(drop=True)
        unlabeled_df = full_df[full_df["label"] == ""].reset_index(drop=True)
        return labeled_df, unlabeled_df
    else:
        return full_df
