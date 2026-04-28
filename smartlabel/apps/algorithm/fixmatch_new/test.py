import argparse
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return argparse.Namespace(**config)


def load_model(args):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(
            depth=args.model_depth,
            widen_factor=args.model_width,
            dropout=0,
            num_classes=args.num_classes
        )
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(
            cardinality=args.model_cardinality,
            depth=args.model_depth,
            width=args.model_width,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    return model


def load_label_map(image_dir):
    label_map_path = os.path.join(os.path.dirname(image_dir), "label2id.json")
    if not os.path.exists(label_map_path):
        print(
            f"label2id.json not found at {label_map_path}, using numeric labels.")
        return None
    with open(label_map_path, 'r') as f:
        label2id = json.load(f)
    # 反转字典：从 id -> label
    id2label = {v: k for k, v in label2id.items()}
    return id2label


def predict(img_path, model, device, num_classes):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return pred.item(), conf.item(), probs.squeeze().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./config.json', help='Path to config.json')
    parser.add_argument('--checkpoint', type=str, default='./result/model_best.pth.tar',
                        help='Path to model .tar or .pth file')
    parser.add_argument(
        '--image', type=str, default='./image_0_label_frog.png', help='Path to image file')
    parser.add_argument('--label_map', type=str,
                        default='./result/label2id.json', help='Path to label2id.json')
    args_cmd = parser.parse_args()
    # 1. 加载配置文件
    args = parse_config(args_cmd.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载模型
    model = load_model(args)
    model.to(device)

    # 3. 加载 checkpoint
    checkpoint = torch.load(args_cmd.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # 4. 加载 label 映射
    id2label = load_label_map(args_cmd.label_map)

    # 5. 预测图像
    pred_class, confidence, prob_list = predict(
        args_cmd.image, model, device, args.num_classes)

    label_name = id2label[pred_class] if id2label else str(pred_class)

    print(f"{label_name}")
    print(f"    {confidence:.4f}")
    # print(f"Probabilities: {['{:.4f}'.format(p) for p in prob_list]}")


if __name__ == '__main__':
    main()
