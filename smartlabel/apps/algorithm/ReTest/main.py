import os
import logging
import json
import argparse
from typing import Dict, Any, Tuple, cast
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import pandas as pd
import sys

from utils.data_loader import CustomImageDataset, get_dataloader
from utils.transform import train_transform, eval_transform
from utils.logger import setup_color_logger
from VolMinNet import estimate_transition_matrix
from CWD import CWD


def main(config: Dict[str, Any], logger: logging.Logger) -> None:
    logger.info("The Re-Test process is starting...")

    # step 1: Load data
    dataset_config = config["dataset_config"]
    volminnet_config = config["volminnet_config"]
    CWD_config = config["CWD_config"]
    train_loader: DataLoader[Tuple[torch.Tensor, int, str, str]] = get_dataloader(dataset_config, batch_size=dataset_config.get("batch_size", 64), transform=train_transform())
    eval_loader: DataLoader[Tuple[torch.Tensor, int, str, str]] = get_dataloader(dataset_config, batch_size=dataset_config.get("batch_size", 64), shuffle=False, transform=eval_transform())

    dataset = cast(CustomImageDataset, train_loader.dataset)
    num_classes = dataset.num_classes

    logger.debug(f"Loaded dataset with {len(dataset)} samples.")
    logger.debug(f"Number of classes: {dataset.num_classes}")
    logger.debug(f"Output Path: {dataset.output_path}")
    logger.debug(f"Index-to-class mapping: {dataset.idx_to_class}")
    logger.debug(f"Class-to-index mapping: {dataset.class_to_idx}") 

    # step 2: Calculate transsition matrix T using VolMinNet
    logger.info("Start calculating transition matrix T...")
    T_hat = estimate_transition_matrix(volminnet_config, dataset, num_classes, logger)
    # logger.debug(f"Estimated Transition Matrix: {T_hat}")
    # logger.debug(f"Transition Matrix Shape: {T_hat.shape}")
    # sys.exit(0)
    # T_hat = [
    #     [0.9662, 0.0010, 0.0060, 0.0040, 0.0020, 0.0010, 0.0040, 0.0000, 0.0119, 0.0040],
    #     [0.0021, 0.9733, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0041, 0.0205],
    #     [0.0097, 0.0000, 0.8876, 0.0126, 0.0378, 0.0029, 0.0446, 0.0029, 0.0019, 0.0000],
    #     [0.0010, 0.0000, 0.0089, 0.8563, 0.0266, 0.0571, 0.0463, 0.0010, 0.0030, 0.0000],
    #     [0.0020, 0.0000, 0.0250, 0.0140, 0.8989, 0.0100, 0.0270, 0.0220, 0.0010, 0.0000],
    #     [0.0000, 0.0000, 0.0085, 0.0790, 0.0213, 0.8698, 0.0064, 0.0139, 0.0011, 0.0000],
    #     [0.0058, 0.0000, 0.0117, 0.0223, 0.0155, 0.0039, 0.9398, 0.0000, 0.0010, 0.0000],
    #     [0.0030, 0.0000, 0.0070, 0.0060, 0.0220, 0.0110, 0.0030, 0.9461, 0.0010, 0.0010],
    #     [0.0195, 0.0029, 0.0010, 0.0049, 0.0020, 0.0010, 0.0049, 0.0000, 0.9580, 0.0059],
    #     [0.0133, 0.0133, 0.0010, 0.0000, 0.0020, 0.0010, 0.0010, 0.0000, 0.0051, 0.9633],
    # ]

    # step 3: Re-Test the dataset uing Class-Wise Denoising
    logger.info("Start the CWD process...")
    model = CWD(CWD_config, train_loader, dataset, T_hat, logger)
    # logger.debug(f"Re-Tested Model: {model}")

    # step 4: Output the final results
    logger.info("Start outputting the final results...")
    # model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    T_tensor = torch.tensor(T_hat, dtype=torch.float32, device=device)

    output_path = dataset.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_preds = []
    image_names = []
    with torch.no_grad():
        for x, _, name, _ in eval_loader:
            x = x.to(device)
            logits = model(x)
            # logger.debug(f"logits.mean(): {logits.mean().item()}")
            if isinstance(logits, tuple):
                logits = logits[0]

            # probs_clean = F.softmax(logits, dim=1)
            # probs_noisy = probs_clean @ T_tensor
            # pred = torch.argmax(probs_noisy, dim=1)

            probs_clean = F.softmax(logits, dim=1)
            pred = torch.argmax(probs_clean, dim=1)

            all_preds.extend(pred.cpu().numpy())
            image_names.extend(name)

    idx_to_class = dataset.idx_to_class
    pred_str_labels = [idx_to_class[idx] for idx in all_preds]

    pred_df = pd.DataFrame({
        dataset.image_name_col: image_names,
        "label2": pred_str_labels
    })

    df = dataset.data.copy()
    out = df.merge(pred_df, on=dataset.image_name_col, how="left")
    out[[dataset.image_name_col, dataset.label_col, "label2"]].to_csv(output_path, index=False)
    logger.info(f"Repredicted label has been stored to: {output_path}")

    logger.info("The Re-Test process is done...")


if __name__ == "__main__":
    logger = setup_color_logger()

    current_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Predict the true label of the given data under label noise.")
    parser.add_argument("--config", type=str, default=os.path.join(current_path, "config", "text.json"), help="Path to the config file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    logger.debug(f"Config: {config}")
    main(config, logger)
