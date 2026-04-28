import json
import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_from_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 从配置读取路径
    true_labels_path = config.get("eval_true_path", "data/pure.csv")
    pred_labels_path = config.get("output_path", "outputs/") + "/result.csv"
    output_path = config.get("eval_output_path", "outputs/merged_result.csv")

    print("开始评估...")
    df_true = pd.read_csv(true_labels_path)
    df_pred = pd.read_csv(pred_labels_path)

    # 避免列名冲突
    df_true = df_true.rename(columns={"label": "label_true"})
    df_pred = df_pred.rename(columns={"label": "label_pred"})

    # 去重保证 text 唯一
    if not df_true['text'].is_unique or not df_pred['text'].is_unique:
        df_true = df_true.drop_duplicates(subset='text')
        df_pred = df_pred.drop_duplicates(subset='text')

    # 合并
    df_merged = pd.merge(df_true, df_pred, on="text", how="inner")
    print(f"原始行数: true={len(df_true)}, pred={len(df_pred)}, 合并后={len(df_merged)}")

    # 准确率评估
    acc_vote = accuracy_score(df_merged["label_true"], df_merged["label_pred"])
    print(f"最终投票预测准确率: {acc_vote * 100:.2f}%")

    if config.get("cg3_use", True) and "cg3_pred" in df_merged.columns:
        acc_cg3 = accuracy_score(df_merged["label_true"], df_merged["cg3_pred"])
        print(f"CG3 预测准确率: {acc_cg3 * 100:.2f}%")

    if config.get("llgc_use", True) and "llgc_pred" in df_merged.columns:
        acc_llgc = accuracy_score(df_merged["label_true"], df_merged["llgc_pred"])
        print(f"LLGC 预测准确率: {acc_llgc * 100:.2f}%")

    if config.get("manifold_use", True) and "manifold_pred" in df_merged.columns:
        acc_mani = accuracy_score(df_merged["label_true"], df_merged["manifold_pred"])
        print(f"Manifold 预测准确率: {acc_mani * 100:.2f}%")

    if config.get("mixtext_use", True) and "mixtext_pred" in df_merged.columns:
        acc_mani = accuracy_score(df_merged["label_true"], df_merged["mixtext_pred"])
        print(f"MixText 预测准确率: {acc_mani * 100:.2f}%")

    # 保存合并结果
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"合并结果已保存到: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    evaluate_from_config(args.config)
