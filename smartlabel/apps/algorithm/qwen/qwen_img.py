# smartlabel/apps/algorithm/qwen/qwen_img.py

import argparse
import json
import re
import sys
import torch
import os
import csv
from PIL import Image
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# --- PARSER MODIFICATION ---
def parse_label_and_confidence(response: str) -> (str, float):
    """
    一个更健壮的函数，用于从模型的混乱响应中解析标签和置信度。
    V2版本增强了对干扰词的过滤和对标签的筛选。

    Args:
        response: 模型返回的原始字符串。

    Returns:
        一个元组 (label, confidence)。
    """
    # 1. 初始清理：移除Markdown、标签、各种符号和换行符
    text = response.strip()
    text = re.sub(r'\*\*|<ref>.*?</ref>|<box>.*?</box>|\[|\]|【|】|“|”|\'|"|\n', '', text)
    text = text.replace('。', '').replace('，', ' ') # 将中文逗号和句号替换为空格
    
    label = "解析失败"
    confidence = 0.0

    # 2. 策略一：直接匹配最理想的格式 "标签 空格 数值"
    # 这个正则表达式寻找一个或多个中文字符/英文字母，后跟一个0-1之间的小数
    match = re.search(r'([\u4e00-\u9fa5a-zA-Z]+)\s+([01]\.?\d+)', text)
    if match:
        label = match.group(1).strip()
        try:
            confidence = float(match.group(2))
            # 增加一个检查，防止"置信度"等词被识别为标签
            if '置信度' not in label and '标签' not in label:
                 return label, confidence
        except (ValueError, TypeError):
            pass # 如果转换失败，继续尝试其他策略

    # 3. 策略二：处理复杂的描述性句子
    # 这是一个更强大的回退策略，当理想格式不存在时使用
    
    # 3.1 首先，提取句子中所有可能的数值作为置信度
    confidence_matches = re.findall(r'([01]\.?\d+)', text)
    if confidence_matches:
        try:
            # 通常最相关的数值是最后一个
            confidence = float(confidence_matches[-1])
        except (ValueError, TypeError):
            confidence = 0.0
            
    # 3.2 其次，定义一个包含所有已知干扰词的“黑名单”
    # 这些词绝不应该成为最终的标签
    blacklist = [
        '置信度', '置信', '度为', '约为', '标签', '是', '的', '一张', '一个', '这', 
        '图像', '图片', '照片', '图中', '主体', '内容', '显示', '归类', '分类',
        '具有', '很高', '较高', '可能', '是', '解析失败'
    ]

    # 3.3 提取所有可能的文本片段，并用黑名单进行过滤
    # re.split 使用数字和空格作为分隔符，将句子拆分成文本块
    possible_labels = re.split(r'\s|[0-9.]+', text)
    
    clean_labels = []
    for p_label in possible_labels:
        p_label = p_label.strip()
        if not p_label:  # 跳过空字符串
            continue
        
        is_bad = False
        for bad_word in blacklist:
            if bad_word in p_label:
                is_bad = True
                break
        
        if not is_bad:
            clean_labels.append(p_label)

    # 3.4 从过滤后的干净标签中选择最可能的一个
    if clean_labels:
        # 通常最核心的名词在句子的最前面，所以我们选择第一个非黑名单词
        label = clean_labels[0]

    # 4. 最后的补救措施
    # 如果经过以上步骤仍然没有找到标签，但找到了置信度，并且原文很短
    if label == "解析失败" and confidence > 0 and len(text.split()) <= 3:
        # 移除所有数字和点，剩下的部分可能就是标签
        potential_label = re.sub(r'[0-9\s\.]+', '', text).strip()
        if potential_label and all(bad not in potential_label for bad in blacklist):
             label = potential_label

    # 5. 最终验证
    if label == "解析失败" or len(label) > 10: # 如果标签过长，也认为是失败
        return "解析失败", 0.0

    return label, confidence


# --- 解析命令行参数 ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/config.json",
    help="Path to config JSON file"
)
args = parser.parse_args()

# --- 读取配置 ---
with open(args.config, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 获取配置参数
image_path_config = config["image_path"]
generation_params = config.get("generation_params", {})
output_path = config.get("output_path", "results.csv")

# 获取图像列表
image_files = []
if isinstance(image_path_config, list):
    for path in image_path_config:
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_files.append(path)
elif isinstance(image_path_config, str):
    if os.path.isfile(image_path_config):
        if image_path_config.endswith('.txt'):
            with open(image_path_config, 'r') as f:
                for line in f:
                    img_file = line.strip()
                    if os.path.isfile(img_file) and img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        image_files.append(img_file)
        elif image_path_config.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_files.append(image_path_config)
    elif os.path.isdir(image_path_config):
        for root, _, files in os.walk(image_path_config):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_files.append(os.path.join(root, file))
else:
    raise TypeError(f"image_path的类型不正确: {type(image_path_config)}")

if not image_files:
    raise ValueError("未找到任何图像文件")
print(f"找到 {len(image_files)} 个图像文件")


# --- 加载模型和 tokenizer ---
model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.0.0'
model_dir = snapshot_download(model_id, revision=revision)
torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = model_dir
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
for k, v in generation_params.items():
    setattr(model.generation_config, k, v)

# --- 准备输出CSV文件 ---
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image_path', 'image_name', 'label', 'confidence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # --- 使用开放式分类逻辑 ---
    for idx, img_file in enumerate(image_files):
        try:
            # 打印进度信息，确保celery日志可以看到
            print(f"Processing image: {img_file}")
            # 使用sys.stdout.flush()确保日志立即输出
            sys.stdout.flush()

            # 检查图像文件的有效性
            if not os.path.isfile(img_file):
                print(f"警告: 找不到图像文件 {img_file}", file=sys.stderr)
                writer.writerow({'image_path': img_file, 'image_name': os.path.basename(img_file), 'label': '文件不存在', 'confidence': 0.0})
                continue
            try:
                with Image.open(img_file) as img:
                    pass
            except Exception as e:
                print(f"警告: 无法打开图像文件 {img_file}: {str(e)}", file=sys.stderr)
                writer.writerow({'image_path': img_file, 'image_name': os.path.basename(img_file), 'label': '图像文件无效', 'confidence': 0.0})
                continue
            
            # --- PROMPT MODIFICATION ---
            prompt = f"""<img>{img_file}</img>
            你的任务是为图像生成一个唯一的、最核心的中文标签和对应的置信度分数。
            你的回答必须且只能遵循 '标签 置信度' 的格式。
            禁止任何其他文字、解释、描述或标点符号。

            示例:
            猫 0.98
            汽车 0.99
            飞机 1.0

            你的回答:"""
            
            # 模型推理
            response, _ = model.chat(tokenizer, query=prompt, history=None)

            # 使用鲁棒性解析函数
            label, confidence_value = parse_label_and_confidence(response)
            
            writer.writerow({
                'image_path': img_file,
                'image_name': os.path.basename(img_file),
                'label': label,
                'confidence': confidence_value
            })
            
        except Exception as e:
            writer.writerow({
                'image_path': img_file,
                'image_name': os.path.basename(img_file),
                'label': '处理错误',
                'confidence': 0.0
            })
            print(f"处理图像 {img_file} 时发生未知错误: {e}", file=sys.stderr)
            
print(f"所有图像处理完成，结果已保存到 {output_path}")
