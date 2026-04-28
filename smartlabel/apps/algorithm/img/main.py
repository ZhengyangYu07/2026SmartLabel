import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from core.data_manager import SimpleDataManager
from utils.data_utils import inspect_data_structure
from core.uncertainty_calculator import UncertaintyCalculator
import os
import sys
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

print(f"当前工作目录: {os.getcwd()}")
print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")


def create_numeric_labels_csv(original_csv, output_csv):
    """创建带有数字标签的新CSV文件"""
    try:
        # 读取原始CSV
        df = pd.read_csv(original_csv)
        print(f"原始数据前5行:\n{df.head()}")

        # 获取唯一的类别名称
        unique_classes = df['class'].unique()
        print(f"发现 {len(unique_classes)} 个唯一类别: {sorted(unique_classes)}")

        # 创建类别到数字的映射
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(unique_classes))}
        print(f"类别映射: {class_to_idx}")

        # 添加数字标签列
        df['class_idx'] = df['class'].map(class_to_idx)

        # 保存新的CSV文件（包含数字标签）
        df[['filename', 'class_idx']].to_csv(output_csv, index=False)
        print(f"已创建数字标签文件: {output_csv}")
        print(f"新文件前5行:\n{df[['filename', 'class_idx']].head()}")

        return class_to_idx

    except Exception as e:
        print(f"创建数字标签CSV失败: {e}")
        return None


def test_with_real_data():
    """使用真实数据测试"""
    print("=== 使用真实数据测试 ===")

    # 路径设置
    base_data_path = os.path.join(os.path.dirname(__file__), "data", "custom")
    image_dir = os.path.join(base_data_path, "images")
    label_csv = os.path.join(base_data_path, "labels.csv")
    numeric_label_csv = os.path.join(base_data_path, "labels_numeric.csv")

    print(f"数据基础路径: {base_data_path}")

    # 检查数据目录结构
    inspect_data_structure(base_data_path)

    # 检查文件是否存在
    print(f"\n检查文件是否存在:")
    print(f"  图片目录: {image_dir} -> {os.path.exists(image_dir)}")
    print(f"  标签文件: {label_csv} -> {os.path.exists(label_csv)}")

    if not os.path.exists(image_dir):
        print(f"错误: 图片目录不存在: {image_dir}")
        return

    # 创建数字标签的CSV文件
    print("\n1. 创建数字标签文件...")
    class_mapping = create_numeric_labels_csv(label_csv, numeric_label_csv)

    if class_mapping is None:
        print("创建数字标签文件失败，尝试使用原始标签文件...")
        numeric_label_csv = label_csv  # 回退到原始文件

    # 初始化数据管理器
    print("\n2. 初始化数据管理器...")
    try:
        data_manager = SimpleDataManager(
            image_dir=image_dir,
            label_csv=numeric_label_csv if os.path.exists(numeric_label_csv) else None
        )
        print("数据管理器初始化成功!")
    except Exception as e:
        print(f"初始化数据管理器失败: {e}")
        print("尝试不使用标签文件...")
        try:
            data_manager = SimpleDataManager(
                image_dir=image_dir,
                label_csv=None  # 不加载标签文件
            )
            print("数据管理器初始化成功（无标签模式）!")
        except Exception as e2:
            print(f"无标签模式也失败: {e2}")
            return

    # 查看统计信息
    print("\n3. 数据统计:")
    try:
        stats = data_manager.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"获取统计信息失败: {e}")

    # 测试数据集
    print("\n4. 测试数据集...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 获取标注数据集
    if hasattr(data_manager, 'labeled_paths') and len(data_manager.labeled_paths) > 0:
        try:
            labeled_dataset = data_manager.get_labeled_dataset(transform)
            print(f"   标注数据集大小: {len(labeled_dataset)}")

            # 显示前几个标注样本（包含类别名称）
            print("   前5个标注样本:")
            for i in range(min(5, len(labeled_dataset))):
                path = labeled_dataset.get_path(i)
                if hasattr(data_manager, 'label_dict') and path in data_manager.label_dict:
                    label_idx = data_manager.label_dict[path]
                    # 尝试获取类别名称
                    label_name = "未知"
                    if class_mapping:
                        for name, idx in class_mapping.items():
                            if idx == label_idx:
                                label_name = name
                                break
                    print(f"     {os.path.basename(path)} -> 类别 {label_idx} ({label_name})")
                else:
                    print(f"     {os.path.basename(path)} -> 无标签信息")
        except Exception as e:
            print(f"处理标注数据集失败: {e}")
    else:
        print("   没有标注数据")

    # 获取未标注数据集
    if hasattr(data_manager, 'unlabeled_paths') and len(data_manager.unlabeled_paths) > 0:
        try:
            unlabeled_dataset = data_manager.get_unlabeled_dataset(transform)
            print(f"   未标注数据集大小: {len(unlabeled_dataset)}")

            # 显示前几个未标注样本
            print("   前3个未标注样本:")
            for i in range(min(3, len(unlabeled_dataset))):
                path = unlabeled_dataset.get_path(i)
                print(f"     {os.path.basename(path)}")
        except Exception as e:
            print(f"处理未标注数据集失败: {e}")
    else:
        print("   没有未标注数据")

    # 测试数据加载
    print("\n5. 测试数据加载...")
    if hasattr(data_manager, 'labeled_paths') and len(data_manager.labeled_paths) > 0:
        try:
            labeled_loader = DataLoader(labeled_dataset, batch_size=2, shuffle=True)

            for batch_idx, (images, labels) in enumerate(labeled_loader):
                print(f"   Batch {batch_idx}: images {images.shape}, labels {labels}")
                # 显示标签对应的类别名称
                if class_mapping and batch_idx == 0:
                    label_names = []
                    for label in labels:
                        for name, idx in class_mapping.items():
                            if idx == label.item():
                                label_names.append(name)
                                break
                    print(f"     标签对应类别: {label_names}")
                if batch_idx == 1:  # 只看前两个batch
                    break
        except Exception as e:
            print(f"数据加载失败: {e}")
    # 6. 测试不确定性计算
    print("\n6. 测试不确定性计算...")
    if len(data_manager.unlabeled_paths) > 0:
        try:
            # 初始化不确定性计算器
            uncertainty_calc = UncertaintyCalculator()

            # 创建未标注数据加载器
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=False)

            # 创建简单模型用于测试（这里用随机模型模拟）
            class SimpleModel(torch.nn.Module):
                def __init__(self, num_classes=10):
                    super().__init__()
                    self.classifier = torch.nn.Linear(3 * 224 * 224, num_classes)

                def forward(self, x):
                    x = x.view(x.size(0), -1)
                    return self.classifier(x)

            model = SimpleModel(num_classes=10)
            model.eval()

            # 测试不同的不确定性计算方法
            methods = ["entropy", "least_confidence", "margin"]

            for method in methods:
                print(f"   使用 {method} 方法计算不确定性...")
                uncertainties = uncertainty_calc.calculate_uncertainty(
                    model=model,
                    dataloader=unlabeled_loader,
                    method=method
                )

                if len(uncertainties) > 0:
                    print(f"     计算了 {len(uncertainties)} 个样本的不确定性")
                    print(f"     不确定性范围: {min(uncertainties):.4f} - {max(uncertainties):.4f}")
                    print(f"     平均不确定性: {sum(uncertainties) / len(uncertainties):.4f}")

                    # 选择最不确定的样本
                    top_indices = uncertainty_calc.select_most_uncertain(uncertainties, top_k=10)
                    print(f"     最不确定的10个样本索引: {top_indices}")

        except Exception as e:
            print(f"   不确定性计算失败: {e}")

    print("\n=== 真实数据测试完成 ===")



def main():
    # 测试真实数据
    test_with_real_data()


if __name__ == "__main__":
    main()