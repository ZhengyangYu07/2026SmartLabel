"""
上传标签文件的解析工具：把用户上传的标签文件转成 {文件名: 标签} 字典。

用于半监督模式下读取初始标注，供后续与数据文件比对、抽取类别列表。
"""
import csv
import json
import os


def parse_label_file(file_obj) -> dict:
    """
    解析图像分类任务中的标签文件，返回一个字典：{文件名: 标签}
    支持的文件格式：.csv, .txt, .json（每行一个样本）

    :param file_obj: Django InMemoryUploadedFile 或 File 对象
    :return: dict[str, str]
    """
    filename = file_obj.name.lower()
    extension = os.path.splitext(filename)[1]

    label_map = {}

    # 读取文件内容
    content = file_obj.read().decode('utf-8').strip()
    lines = content.splitlines()

    if extension == '.csv':
        # CSV：取前两列，去掉包裹的引号
        reader = csv.reader(lines)
        for row in reader:
            if len(row) >= 2:
                image_name = row[0].strip().strip('"')
                label = row[1].strip().strip('"')
                label_map[image_name] = label

    elif extension == '.txt':
        # TXT：每行 "文件名,标签"，只按第一个逗号切分（标签中可能含逗号）
        for line in lines:
            if ',' in line:
                image_name, label = line.split(',', 1)
                label_map[image_name.strip()] = label.strip()

    elif extension == '.json':
        # JSON：每行一个对象，字段为 filename / label
        for line in lines:
            try:
                obj = json.loads(line)
                image_name = obj.get("filename")
                label = obj.get("label")
                if image_name and label:
                    label_map[image_name.strip()] = label.strip()
            except json.JSONDecodeError as e:
                raise ValueError(f"无效的 JSON 行: {line}")

    else:
        raise ValueError(f"不支持的文件格式: {extension}")

    return label_map
