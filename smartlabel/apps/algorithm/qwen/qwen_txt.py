import argparse
import json
import re
import sys
import torch
import os
import csv
from pathlib import Path
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# --- PARSER MODIFICATION ---
def parse_label_and_confidence(response: str) -> (str, float):
    """
    Args:
        response: 模型返回的原始字符串。

    Returns:
        一个元组 (label, confidence)。
    """    
    label = "解析失败"
    confidence = 0.0

    # 0. 定义一个标签黑名单。**重要：请根据你的实际目标分类标签来调整此列表！**
    # 例如，如果“新闻”、“体育”是你的合法标签，则请从这个列表中移除它们。
    # 此列表旨在过滤掉模型可能误生成的引导词、无意义片段或系统性词汇。
    strict_blacklist = [
        '置信度', '置信', '度为', '约为', '标签', '是', '的', '一个', '这', '该',
        '文本', '主题', '内容', '显示', '归类', '分类', '关于', '一篇', '属于',
        '具有', '很高', '较高', '可能', '我', '认为', '解析失败',
        # 常见数字和货币符号，防止模型将数字识别为标签
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '十', '百', '千', '万', '亿',
        '美元', '元', '日元', '欧元', '英镑', '卢布', '人民币', '¥', '$', '€', '£',
        # 常见无意义的英文断词或介词/冠词，确保标签的完整性
        'a', 'an', 'the', 'in', 'on', 'at', 'of', 'for', 'with', 'and', 'or', 'not',
        'is', 'am', 'are', 'was', 'were', 'be', 'to', 'from', 'by', 'as', 'it', 'its', 'he', 'she', 'they',
        'ing', 'ed', 'er', 'est', # 常见英文后缀，通常不作为独立标签
    ]
    # 将黑名单转换为小写，方便不区分大小写的匹配
    strict_blacklist_lower = [w.lower() for w in strict_blacklist]

    # 1. 初始清理和标准化：更温和地移除 Markdown、多余空格和常见引导词。
    # 保留连字符 '-' 和下划线 '_'，因为它们可能是标签的一部分。
    text = response.strip()
    # 移除Markdown、HTML标签、特殊括号、引号、换行符、反引号等，**但保留连字符'-'和下划线'_'**
    text = re.sub(r'\*\*|<[^>]+>|\[|\]|【|】|“|”|\'|"|\n|`|---|\||#|\(|\)|\{|\}|<|>|：|,|。', ' ', text)
    # 移除可能出现在标签前的引导词
    text = re.sub(r'(?:标签|分类|主题|是|为|置信度)\s*', ' ', text, flags=re.IGNORECASE)
    # 标准化所有连续的空白字符为一个空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. 核心策略：寻找 "标签 + [0,1]范围内的置信度" 的组合
    # 标签部分：放宽长度限制（1-30字符），允许中文、英文、数字、连字符、下划线。
    # 匹配模式：[中文/英文/数字/连字符/下划线组合，长度1-30] + 空格 + [置信度]
    # 使用 \b 确保匹配的是完整词边界，避免从长词中截取部分作为标签（例如从"running"中匹配"run"）
    # 但对于中文词，\b 的效果可能不理想，所以对于中文标签，主要靠长度和黑名单来过滤
    pattern_label_conf = re.compile(r'(\b[\u4e00-\u9fa5a-zA-Z0-9\-_]{2,30}\b)\s*(0\.\d+|1\.0|1|0)')
    matches = pattern_label_conf.findall(text)

    if matches:
        # 从后往前遍历，优先选择最靠近文本末尾的有效匹配，通常模型会把最终结论放在后面
        for l_candidate, c_str in reversed(matches):
            l_candidate = l_candidate.strip()
            
            # 过滤掉纯数字的标签
            if l_candidate.isdigit():
                continue

            # 过滤掉黑名单中的词（不区分大小写）
            if l_candidate.lower() in strict_blacklist_lower:
                continue
            
            # 确保标签包含至少一个中文、英文字母或数字，避免纯标点符号作为标签
            if not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', l_candidate):
                 continue

            # 强制标签最小长度：1个字符对于中文有时是词，但对于分类标签很少见。
            # 通常分类标签至少2个字符，可以有效过滤“A”、“s”之类的无意义单字母。
            # 如果你的业务场景确实有单字中文标签，请将此值设为1。
            if len(l_candidate) < 2:
                continue

            try:
                conf = float(c_str)
                if 0.0 <= conf <= 1.0:
                    label = l_candidate
                    confidence = conf
                    break # 找到第一个符合条件的就停止
            except ValueError:
                continue

    # 3. 如果核心策略未能找到，尝试更宽松的策略：先提取置信度，再在附近寻找标签
    if label == "解析失败":
        confidence_matches = re.findall(r'(0\.\d+|1\.0|1|0)', text)
        if confidence_matches:
            try:
                potential_confidence = float(confidence_matches[-1]) # 同样，取最后一个数字作为置信度
                if 0.0 <= potential_confidence <= 1.0:
                    # 获取置信度之前的文本，并在其中寻找可能的标签
                    text_before_confidence = text.rsplit(confidence_matches[-1], 1)[0]
                    
                    # 进一步清理置信度之前的文本，只保留可能作为标签的字符
                    # 允许中文、英文、数字、连字符、下划线，移除其他标点和多余空格
                    cleaned_text_before = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\-_]', ' ', text_before_confidence)
                    cleaned_text_before = re.sub(r'\s+', ' ', cleaned_text_before).strip()
                    
                    # 从清理后的文本中分割出单词/短语
                    possible_labels = [p.strip() for p in cleaned_text_before.split(' ') if p.strip()]
                    
                    for p_label in reversed(possible_labels): # 从后往前找最近的词作为标签
                        if not p_label: continue
                        if p_label.isdigit(): continue # 纯数字不是标签
                        if p_label.lower() in strict_blacklist_lower: continue # 严格黑名单过滤
                        if len(p_label) > 30: continue # 标签长度限制
                        if len(p_label) < 2: continue # 强制最小长度
                        if not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', p_label): continue # 必须包含字母或数字

                        # 找到一个符合条件的标签
                        label = p_label
                        confidence = potential_confidence
                        break
            except ValueError:
                pass # 如果转换失败，继续

    # 4. 最终验证和清理
    # 如果标签仍是初始值，或长度异常，或纯数字，则判定为失败
    if label == "解析失败" or len(label) == 0 or len(label) > 30 or label.isdigit():
        return "解析失败", 0.0
    
    # 再次检查标签是否在黑名单中（以防万一）
    if label.lower() in strict_blacklist_lower:
        return "解析失败", 0.0

    # 确保置信度在[0,1]范围内
    if not (0.0 <= confidence <= 1.0):
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

text_input_dir = config["text_input_dir"]
generation_params = config.get("generation_params", {})
output_path = config.get("output_path", "results.csv")

# --- 查找并统计文本数据 (无改动) ---
TEXT_COLUMN_KEYWORDS = ['text', 'content', 'body', 'sentence', 'article', 'description', 'message']
csv_files = list(Path(text_input_dir).glob('*.csv'))
if not csv_files:
    raise FileNotFoundError(f"未在目录 {text_input_dir} 中找到任何CSV文件。")

total_texts = 0
for csv_file_path in csv_files:
    try:
        with open(csv_file_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            is_header_row = False
            if header:
                cleaned_header = [h.strip().lower() for h in header]
                if any(keyword in cleaned_header for keyword in TEXT_COLUMN_KEYWORDS):
                    is_header_row = True
            
            f.seek(0)
            if is_header_row:
                next(reader) 
            
            total_texts += sum(1 for row in reader)
    except Exception as e:
        print(f"警告: 读取文件 {csv_file_path} 以统计行数时出错: {e}", file=sys.stderr)

if total_texts == 0:
    raise ValueError(f"从目录 {text_input_dir} 中的CSV文件未能提取任何文本数据。")
print(f"总计找到 {total_texts} 条文本数据进行处理。")

# --- 加载模型和 tokenizer ---
model_id = config.get('model_id', 'qwen/Qwen-VL-Chat')
revision = config.get('revision', 'v1.0.0')

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
processed_count = 0

with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['text_id', 'content', 'label', 'confidence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # --- 主处理循环 ---
    for csv_file_path in csv_files:
        print(f"正在处理文件: {csv_file_path.name}")
        try:
            with open(csv_file_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                text_column_index = 0
                if header:
                    cleaned_header = [h.strip().lower() for h in header]
                    for keyword in TEXT_COLUMN_KEYWORDS:
                        if keyword in cleaned_header:
                            text_column_index = cleaned_header.index(keyword)
                            break
                else:
                    f.seek(0)

                if header:
                    f.seek(0)
                    next(reader)

                for row_idx, row in enumerate(reader):
                    text_id = f"{csv_file_path.stem}_{row_idx + 1}"
                    if text_column_index < len(row):
                        text_content = str(row[text_column_index]).strip()
                    else:
                        print(f"警告: 在文件 {csv_file_path.name} 的第 {row_idx+2} 行找不到文本列，已跳过。", file=sys.stderr)
                        processed_count += 1
                        continue

                    if not text_content:
                        print(f"警告: 在文件 {csv_file_path.name} 的第 {row_idx+2} 行文本内容为空，已跳过。", file=sys.stderr)
                        processed_count += 1
                        continue

                    try:
                        prompt = f"""请分析以下由'---'分隔的文本内容。
                            ---
                            {text_content}
                            ---
                            你的任务是：
                            1. 为这段文本生成一个最核心、最简洁的分类标签（中文或英文均可）。
                            2. 提供一个介于 0.0 和 1.0 之间的置信度分数。

                            你的回答必须严格遵循 '标签 置信度' 的格式，一个标签和一个数字，中间用空格隔开。
                            禁止任何解释、描述、句子、markdown标记或其他任何多余的文字。

                            正确格式示例:
                            体育 0.98
                            科技 0.99
                            财经 1.0

                            你的回答:"""
                        
                        # 模型推理
                        response, _ = model.chat(tokenizer, query=prompt, history=None)

                        # 使用全新的、健壮的解析函数
                        label, confidence_value = parse_label_and_confidence(response)
                        
                        writer.writerow({
                            'text_id': text_id,
                            'content': text_content,
                            'label': label,
                            'confidence': confidence_value
                        })
                        
                    except Exception as e:
                        writer.writerow({
                            'text_id': text_id,
                            'content': text_content,
                            'label': '处理错误',
                            'confidence': 0.0
                        })
                        print(f"处理文本 ID: {text_id} 时发生错误: {str(e)}", file=sys.stderr)
                    
                    processed_count += 1
                    if total_texts > 0:
                        progress = int((processed_count / total_texts) * 100)
                        if processed_count % 10 == 0 or progress > int(((processed_count - 1) / total_texts) * 100):
                             print(f"处理进度: {progress}% ({processed_count}/{total_texts})", end='\r')
                             sys.stdout.flush()
        except Exception as e:
            print(f"\n处理文件 {csv_file_path.name} 时发生严重错误: {e}", file=sys.stderr)

print(f"\n所有文本处理完成，结果已保存到 {output_path}")
