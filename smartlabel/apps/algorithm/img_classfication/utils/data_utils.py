import pandas as pd
import torch
import os

# def load_image_dataset(image_dir, label_csv, split=True):
#     """
#     Args:
#         image_dir (str): 图像文件所在的目录。
#         label_csv (str): 标签 CSV 文件，需包含 'filename' 和 'class' 两列。
#         split (bool): 是否区分有标签和无标签样本。
#     Returns:
#         labeled_df, unlabeled_df 或 合并后的 df
#     """
#     df = pd.read_csv(label_csv)

#     # 保证字段存在
#     assert "filename" in df.columns and "class" in df.columns, \
#         "label_csv 必须包含 'filename' 和 'class' 两列"

#     # 拼接图像路径
#     df["filepath"] = df["filename"].apply(lambda x: os.path.join(image_dir, x))
#     df["text"] = df["filepath"]  # 兼容旧结构字段名 text
#     df["label"] = df["class"].fillna("")

#     if split:
#         labeled_df = df[df["label"] != ""].reset_index(drop=True)
#         unlabeled_df = df[df["label"] == ""].reset_index(drop=True)
#         return labeled_df, unlabeled_df
#     else:
#         return df

def load_image_dataset(image_dir, label_csv=None, split=True):
    """
    Args:
        image_dir (str): 图像文件所在的目录（会递归搜索其所有子目录）。
        label_csv (str): 标签 CSV 文件（可选），需包含 'filename' 和 'class' 两列。
        split (bool): 是否区分有标签和无标签样本。
    Returns:
        labeled_df, unlabeled_df 或合并后的 df
    """
    # 递归搜索所有图片文件
    all_files = []
    file_paths = []
    
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                # 保存文件名和完整路径
                all_files.append(file)
                file_paths.append(os.path.join(root, file))
    
    all_files_with_paths = list(zip(all_files, file_paths))
    all_files_with_paths.sort(key=lambda x: x[0])  # 按文件名排序
    
    # 如果有 label_csv
    if label_csv is not None and os.path.exists(label_csv):
        df = pd.read_csv(label_csv)
        assert "filename" in df.columns and "class" in df.columns, \
            "label_csv 必须包含 'filename' 和 'class' 两列"
        df["filename"] = df["filename"].astype(str).str.strip()
        labeled_filenames = set(df["filename"])
    else:
        df = pd.DataFrame(columns=["filename", "class"])
        labeled_filenames = set()

    # 有标签样本
    labeled_df = df.copy()
    # 找到每个标记文件对应的真实路径
    filepath_map = {os.path.basename(path): path for _, path in all_files_with_paths}
    labeled_df["filepath"] = labeled_df["filename"].apply(lambda x: filepath_map.get(x, os.path.join(image_dir, x)))
    labeled_df["text"] = labeled_df["filepath"]
    labeled_df["label"] = labeled_df["class"]

    # 找出未出现在 label_csv 里的图像文件
    unlabeled_files = []
    unlabeled_paths = []
    
    for filename, filepath in all_files_with_paths:
        if filename not in labeled_filenames:
            unlabeled_files.append(filename)
            unlabeled_paths.append(filepath)
    
    unlabeled_df = pd.DataFrame({
        "filename": unlabeled_files,
        "class": "",
        "filepath": unlabeled_paths,
        "text": unlabeled_paths,
        "label": ["" for _ in unlabeled_files]
    })

    print(f"[Info] Found {len(labeled_df)} labeled images and {len(unlabeled_df)} unlabeled images")
    
    if split:
        return labeled_df.reset_index(drop=True), unlabeled_df.reset_index(drop=True)
    else:
        return pd.concat([labeled_df, unlabeled_df], ignore_index=True)

def encode_labels(labels):
    label_set = sorted(set(l for l in labels if l != -1))
    label2id = {l: i for i, l in enumerate(label_set)}
    id2label = {i: l for l, i in label2id.items()}
    encoded = torch.LongTensor([label2id[l] if l in label2id else -1 for l in labels])
    return encoded, label2id, id2label

def get_mask(encoded_labels):
    return encoded_labels != -1
