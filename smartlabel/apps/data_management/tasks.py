import csv
import glob
import json
import logging
import math
import os
import pandas as pd
import psutil
import shlex
import stat
import subprocess
import time
import uuid
from datetime import datetime, timezone
from celery import shared_task
from celery.exceptions import Ignore
from django.apps import apps
from django.core.cache import cache
from django.conf import settings
from django.core.exceptions import SuspiciousOperation
from django.db import transaction
from django.db import models
from pathlib import Path
import torch
from smartlabel.apps.algorithm.object_detection.active_learning import detection_uncertainty
from .models import Task

MIN_UPDATE_INTERVAL = 0.5
logger = logging.getLogger(__name__)
DETECTION_PREDICT_CONF_THRES = 0.25
DETECTION_PREDICT_IOU_THRES = 0.45
DETECTION_PREDICT_MAX_DET = 100
DETECTION_MAX_BOXES_PER_IMAGE = 100
DETECTION_PRETRAINED_WEIGHT = Path(os.environ.get(
    'SMARTLABEL_EFFICIENTTEACHER_WEIGHTS',
    str(Path(settings.BASE_DIR) / 'models' / 'efficient-yolov5l-obj365.pt'),
))
DETECTION_LABEL_ALIASES = {
    '船': 'ship',
    '船舶': 'ship',
    '轮船': 'ship',
    '车': 'car',
    '汽车': 'car',
    '车辆': 'car',
    '飞机': 'airplane',
}
DEFAULT_TEXT_CLASSIFICATION_BERT_MODEL = Path(os.environ.get(
    'SMARTLABEL_TEXT_CLASSIFICATION_BERT_MODEL',
    str(Path(settings.BASE_DIR) / 'models' / 'chinese-roberta-wwm-ext'),
))


def _resolve_text_classification_bert_model():
    if DEFAULT_TEXT_CLASSIFICATION_BERT_MODEL.exists():
        return str(DEFAULT_TEXT_CLASSIFICATION_BERT_MODEL)
    return 'hfl/chinese-roberta-wwm-ext'


def _heartbeat_cache_key(task_id):
    return f"task:{task_id}:algo_heartbeat"


def _set_algorithm_heartbeat(task_id, line):
    cache.set(
        _heartbeat_cache_key(task_id),
        {
            'at': datetime.now(timezone.utc).isoformat(),
            'line': (line or '')[:240],
        },
        timeout=24 * 3600,
    )


def clear_algorithm_heartbeat(task_id):
    cache.delete(_heartbeat_cache_key(task_id))


def _update_task_progress(task, progress, celery_task=None):
    """阶段性更新任务进度，避免进度长时间无变化。"""
    progress = int(max(0, min(progress, 100)))
    if task.progress >= progress:
        return

    task.progress = progress
    task.save(update_fields=['progress'])
    if celery_task:
        celery_task.update_state(state='PROGRESS', meta={'progress': progress})

def check_system_resources(task_id=None):
    """检查系统资源使用情况"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()

        task_info = f"[Task-{task_id}] " if task_id else ""
        logger.info(
            f"{task_info}系统资源状态 - "
            f"CPU: {cpu_percent}% "
            f"内存: {mem.percent}% "
            f"可用内存: {mem.available / 1024 / 1024:.2f}MB"
        )

        # 阈值配置
        WARNING_THRESHOLD = 90
        if cpu_percent > WARNING_THRESHOLD:
            logger.warning(f"{task_info}CPU使用率过高: {cpu_percent}%")
        if mem.percent > WARNING_THRESHOLD:
            logger.warning(f"{task_info}内存使用率过高: {mem.percent}%")
            
        return {
            'cpu_percent': cpu_percent,
            'mem_percent': mem.percent,
            'available_mem': mem.available / 1024 / 1024  # MB
        }

    except Exception as e:
        logger.warning(f"{task_info}资源监控失败: {str(e)}")
        return None

def create_progress_tracker(task, task_id, self=None):
    """创建进度跟踪器函数"""
    # 进度跟踪器状态
    progress_data = {
        'current': 0,
        'total': 0,
        'last_updated': time.time()
    }

    def update_progress(current, total):
        nonlocal task
        now = time.time()
        progress = float(current / total * 100) if total > 0 else 0

        # 每当有1%的进度或者规定时间达到时，更新进度
        if (progress - task.progress) >= 1 or \
                (now - progress_data['last_updated']) >= MIN_UPDATE_INTERVAL:
            TaskModel = apps.get_model('data_management.Task')
            with transaction.atomic():
                task = TaskModel.objects.get(id=task_id)
                task.progress = progress
                task.save(update_fields=['progress'])
                if self:  # 如果提供了Celery任务实例
                    self.update_state(
                        state='PROGRESS',
                        meta={'progress': progress}
                    )
                progress_data.update({
                    'last_updated': time.time(),
                    'current': current,
                    'total': total
                })
    return update_progress


def run_subprocess_with_progress(cmd, task_id, total_files=None):
    """
    运行子进程并解析输出中的进度信息，更新任务进度。
    支持两种模式：
    1. 如果提供了 total_files，则会匹配 "正在处理" 等关键词来计算进度。
    2. 如果未提供 total_files，则会匹配 "处理进度: xx%" 来获取进度。
    """
    from celery import current_task
    import re
    import subprocess
    
    try:
        task_obj = Task.objects.get(id=task_id)
        _set_algorithm_heartbeat(task_id, 'subprocess_started')
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            bufsize=1,
            encoding='utf-8', # 明确指定编码
            errors='replace'   # 替换无法解码的字符
        )
        
        # --- 模式B：基于百分比的正则 ---
        progress_pattern = re.compile(r'处理进度: (\d+)%')
        
        # --- 模式A：基于文件计数的正则 ---
        # 匹配 qwen_img.py 可能的输出，例如 "Processing image: xxx.jpg"
        file_progress_pattern = re.compile(r'Processing image:|正在处理图像:') 

        current_progress = 0
        processed_files = 0
        
        # 定义一个标志，指示当前行是否已被“有效”处理过，避免重复记录
        line_processed_as_progress = False

        for line in iter(process.stdout.readline, ''):
            stripped_line = line.strip()
            safe_line = stripped_line.replace('\ufffd', '?')
            if safe_line:
                _set_algorithm_heartbeat(task_id, safe_line)

            # --- 优先匹配模式B (百分比) ---
            match_percent = progress_pattern.search(safe_line)
            if match_percent:
                try:
                    progress = int(match_percent.group(1))
                    progress = max(0, min(progress, 100))
                    
                    if progress > current_progress:
                        current_progress = progress
                        task_obj.progress = progress
                        task_obj.save(update_fields=['progress'])
                        if current_task:
                            current_task.update_state(state='PROGRESS', meta={'progress': progress})
                        logger.info(f"子进程输出 (进度更新): {safe_line}") # 仅在更新进度时记录一次
                        line_processed_as_progress = True # 标记此行已用于进度更新
                except (ValueError, TypeError) as e:
                    logger.error(f"解析进度百分比时出错: {str(e)}")
                continue # 匹配到百分比后，不再进行其他模式匹配

            # --- 如果模式B未匹配，且 total_files 有效，则尝试模式A (文件计数) ---
            if total_files and total_files > 0:
                match_file = file_progress_pattern.search(safe_line)
                if match_file:
                    processed_files += 1
                    progress = int((processed_files / total_files) * 100)
                    progress = max(0, min(progress, 99)) # 文件计数最高到99%，最后由完成逻辑置100
                    
                    if progress > current_progress:
                        current_progress = progress
                        task_obj.progress = progress
                        task_obj.save(update_fields=['progress'])
                        if current_task:
                            current_task.update_state(state='PROGRESS', meta={'progress': progress})
                        logger.info(f"子进程输出 (文件计数更新): {safe_line}") # 仅在更新进度时记录一次
                        line_processed_as_progress = True # 标记此行已用于进度更新
                    continue # 匹配到文件后，不再进行其他模式匹配
                    
            # 如果以上两种进度模式都未匹配，则按通用信息记录一次
            if not line_processed_as_progress:
                logger.info(f"子进程输出: {safe_line}") 
            line_processed_as_progress = False # 重置标志，为下一行准备

        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"子进程返回非零状态码: {return_code}")
            raise subprocess.CalledProcessError(return_code, cmd, output=None, stderr="subprocess exited with non-zero code")
        
        # 确保任务进度在成功后为100%
        if task_obj.progress < 100:
            task_obj.progress = 100
            task_obj.save(update_fields=['progress'])
            if current_task:
                current_task.update_state(state='PROGRESS', meta={'progress': 100})

        _set_algorithm_heartbeat(task_id, 'subprocess_finished')
        
        return return_code
    except Exception as e:
        logger.error(f"执行子进程时出错: {str(e)}")
        _set_algorithm_heartbeat(task_id, f"subprocess_error: {str(e)}")
        if 'process' in locals() and process.poll() is None:
            try:
                process.terminate()
            except: pass
        raise


# 将结果保存到数据库
def save_results_to_database(result_file_path, task, task_type, update_progress_func):
    """保存结果到数据库，并在落库阶段动态推进任务进度。"""
    if not result_file_path.exists():
        logger.error(f"未找到结果文件: {result_file_path}")
        raise FileNotFoundError(f"未找到结果文件: {result_file_path}")

    model_key = {
        'image-classification': 'ImageResult',
        'text-classification': 'TextResult'
    }
    ResultModel = apps.get_model('data_management', model_key[task_type])

    logger.info(f"正在为任务 {task.id} 清除旧的结果记录...")
    try:
        with transaction.atomic():
            deleted_count, _ = ResultModel.objects.filter(task=task).delete()
            logger.info(f"为任务 {task.id} 成功删除了 {deleted_count} 条旧记录。")
    except Exception as e:
        logger.error(f"清除任务 {task.id} 的旧记录时失败: {e}")
        raise  # 如果删除失败，则终止操作

    def _safe_float(value, default=0.0):
        try:
            if value in (None, ''):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    progress_start = 85
    progress_end = 99

    with open(result_file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results_data = list(reader)  # 先读取所有数据到列表

        if not results_data:
            _update_task_progress(task, progress_end)
            return

        # 让落库阶段按数据量动态推进，而不是在最后一次性跳到 100%。
        batch_size = max(1, (len(results_data) + 19) // 20)

        instances_to_create = []
        for i, res_data in enumerate(results_data):
            # 2. 序号直接从1开始，基于文件中的行索引 (i)
            current_sequence_number = i + 1
            
            # 根据任务类型创建模型实例
            if task_type == 'image-classification':
                instance = ResultModel(
                    task=task,
                    sequence_number=current_sequence_number,  # 使用新的从1开始的序号
                    image_name=res_data.get('image_name', ''),
                    image_path=res_data.get('image_path', ''),
                    label=res_data.get('label', ''),
                    confidence=_safe_float(res_data.get('confidence', 0.0))
                )
            elif task_type == 'text-classification':
                instance = ResultModel(
                    task=task,
                    sequence_number=current_sequence_number,  # 使用新的从1开始的序号
                    text_id=res_data.get('text_id', ''),
                    content=res_data.get('content', ''),
                    label=res_data.get('label', ''),
                    confidence=_safe_float(res_data.get('confidence', 0.0))
                )
            else:
                continue # 如果有其他类型，暂时跳过
                
            instances_to_create.append(instance)
            
            # 每达到batch_size或到列表末尾时进行批量创建
            if (i + 1) % batch_size == 0 or (i + 1) == len(results_data):
                with transaction.atomic():  # 确保批次操作的原子性
                    ResultModel.objects.bulk_create(instances_to_create)

                stage_progress = progress_start + round(
                    ((i + 1) / len(results_data)) * (progress_end - progress_start)
                )
                _update_task_progress(task, stage_progress)
                instances_to_create = []  # 清空列表，准备下一个批次

        # 落库阶段结束时，保留最后一点给任务完成状态。
        _update_task_progress(task, progress_end)


# 辅助函数，用于处理通用的任务逻辑和子进程调用
def _process_common_pretrained_task(self, task_id, task, update_progress, result_dir, extract_dir, output_csv, config_path, result_file_path):
    """
    通用逻辑，用于处理预训练模式下的图像和文本分类。
    此函数仅为内部调用，不作为 Celery task。
    """
    cmd = []
    total_input_items = 0
    
    if task.task_type == 'image-classification':
        # 构建图像分类的配置
        config = {
            "image_path": extract_dir,
            "label_choices": [],
            "restrict_labels": True,
            "output_path": output_csv,
            "generation_params": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 512
            }
        }
        # 获取图片文件数量，用于进度跟踪
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    total_input_items += 1
        
        # 构建命令
        cmd = [
            'conda', 'run', '--no-capture-output', '-n', 'qwen', 'python',
            os.path.join(settings.BASE_DIR, 'smartlabel', 'apps', 'algorithm', 'qwen', 'qwen_img.py'),
            '--config', config_path
        ]
        
    elif task.task_type == 'text-classification':
        # 定义常见文本列名关键词
        TEXT_COLUMN_KEYWORDS = ['text', 'content', 'body', 'sentence', 'article', 'description', 'message']
        
        # 统计所有CSV文件中的文本条目总数，用于进度跟踪
        total_input_items = 0
        csv_files = list(Path(extract_dir).glob('*.csv'))

        if not csv_files:
            raise FileNotFoundError(f"文本分类任务未在目录 {extract_dir} 中找到任何CSV文件。")

        for csv_file_path in csv_files:
            logger.info(f"正在统计文件: {csv_file_path}")
            with open(csv_file_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None) # 读取第一行，可能是标题
                
                is_header_row = False
                if header:
                    cleaned_header = [h.strip().lower() for h in header]
                    for keyword in TEXT_COLUMN_KEYWORDS:
                        if keyword in cleaned_header:
                            is_header_row = True
                            break
                
                # 如果第一行是数据行，则算入统计
                if not is_header_row and header:
                    total_input_items += 1 # 统计第一行作为数据

                # 统计剩余行
                for _ in reader:
                    total_input_items += 1
        
        if total_input_items == 0:
            raise ValueError(f"从目录 {extract_dir} 中的CSV文件未能提取任何文本数据。")

        logger.info(f"总计找到 {total_input_items} 条文本数据待处理。")

        # 预训练模式下，不再根据分类场景设置标签列表
        label_choices = []
        
        config = {
            "text_input_dir": str(extract_dir),
            "dataset_path": str(extract_dir),
            "output_path": output_csv,
            "label_choices": label_choices,
            "bert_model": "bert-base-chinese",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 6,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "max_length": 128,
            "hidden_dim": 256
        }

        # 全部使用 BERT
        cmd = [
            'conda', 'run', '--no-capture-output', '-n', 'bert', 'python',
            os.path.join(settings.BASE_DIR, 'smartlabel', 'apps', 'algorithm', 'bert', 'train.py'),
            '--config', config_path
        ]
        
    else:
        raise ValueError(f"未知或不支持的任务类型: {task.task_type}")

    # 保存配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    # 记录命令到日志
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    # 运行命令
    run_subprocess_with_progress(cmd, task_id, total_files=total_input_items)
    
    # 检查结果文件是否存在
    result_path = Path(result_file_path)
    if result_path.exists():
        # 保存结果到数据库
        save_results_to_database(result_path, task, task.task_type, update_progress)
    else:
        raise FileNotFoundError(f"{task.task_type} 结果文件不存在: {result_file_path}")


def _merge_manual_labels_if_needed(task):
    """如果存在人工标注文件，则与原始标注文件合并后返回新的 CSV 路径。"""
    manual_label_path = Path(settings.MEDIA_ROOT) / 'results' / str(task.id) / 'manual_labels.csv'
    original_label_path = None

    if task.label_file and Path(task.label_file.path).exists():
        original_label_path = Path(task.label_file.path)

    if original_label_path is None and not manual_label_path.exists():
        raise FileNotFoundError("请先在人工标注界面保存至少一条标注结果，再重新调用半监督算法。")

    if original_label_path is None:
        return manual_label_path

    if not manual_label_path.exists():
        return original_label_path

    merged_label_path = Path(settings.MEDIA_ROOT) / 'results' / str(task.id) / 'merged_manual_labels.csv'
    merged_label_path.parent.mkdir(parents=True, exist_ok=True)

    merged_records = {}

    for source_path in [original_label_path, manual_label_path]:
        with open(source_path, newline='', encoding='utf-8') as source_file:
            reader = csv.DictReader(source_file)
            for row in reader:
                filename = (row.get('filename') or '').strip()
                label = (row.get('class') or '').strip()
                if filename:
                    merged_records[filename] = label

    with open(merged_label_path, 'w', newline='', encoding='utf-8') as target_file:
        writer = csv.writer(target_file)
        writer.writerow(['filename', 'class'])
        for filename, label in sorted(merged_records.items()):
            writer.writerow([filename, label])

    return merged_label_path


def _manual_text_label_path(task):
    return Path(settings.MEDIA_ROOT) / 'results' / str(task.id) / 'manual_text_labels.csv'


@shared_task(
    bind=True,
    time_limit=7200,
    soft_time_limit=3600
)
def process_pretrained_task(self, task_id, *args, **kwargs):
    """处理预训练任务"""
    logger.info(f"开始处理预训练任务 {task_id}")
    
    try:
        # 获取任务对象
        task = Task.objects.get(id=task_id)
        task.status = 'processing'
        if task.progress < 1:
            task.progress = 1
        task.celery_task_id = self.request.id
        task.save(update_fields=['status', 'progress', 'celery_task_id'])

        # 创建进度跟踪器
        update_progress = create_progress_tracker(task, task_id, self)

        # 创建结果目录
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results', str(task_id))
        os.makedirs(result_dir, exist_ok=True)
        
        # 输入数据集
        extract_dir = os.path.join(settings.MEDIA_ROOT, 'extracted', str(task_id))

        # 输出CSV文件路径
        output_csv = os.path.join(result_dir, 'result.csv')
        
        # 配置文件路径
        config_path = os.path.join(result_dir, 'config.json')
        result_file_path = os.path.join(result_dir, 'result.csv')
        
        # 调用通用处理函数
        _process_common_pretrained_task(self, task_id, task, update_progress, result_dir, extract_dir, output_csv, config_path, result_file_path)
        
        # 更新任务状态
        task.progress = 100 # 确保进度是100
        task.status = 'completed'
        task.save()
        
        logger.info(f"预训练任务 {task_id} 处理完成")
        
    except Task.DoesNotExist:
        logger.error(f"任务 {task_id} 不存在")
        # 直接忽略不存在任务，避免向结果后端写入不完整异常结构触发 exc_type 反序列化错误。
        raise Ignore()
    except Exception as e:
        logger.error(f"处理预训练任务 {task_id} 时出错: {str(e)}")
        
        # 更新任务状态为失败 - 修复错误字段问题
        try:
            task = Task.objects.get(id=task_id)
            task.status = 'failed'
            if hasattr(task, 'error_message'): # 确保 Task 模型有这个字段
                task.error_message = str(e)
                task.save(update_fields=['status', 'error_message'])
            elif hasattr(task, 'error_log'): # 如果是 error_log 字段
                task.error_log = str(e)
                task.save(update_fields=['status', 'error_log'])
            else:
                task.save(update_fields=['status'])
                logger.error(f"任务失败原因: {str(e)}")
        except Exception as update_error:
            logger.error(f"更新任务状态失败: {str(update_error)}")

        # 训练错误通常是确定性的，不再自动重试，避免 worker 反复重复运行同一任务。
        raise Ignore()


@shared_task(
    bind=True,
    time_limit=7200,
    soft_time_limit=3600
)
def process_annotation_task(self, task_id):
    """处理半监督模式的任务 (修复 'glob' 错误并增加健壮性)"""
    logger.info(f"开始处理半监督任务: {task_id}")
    check_system_resources(task_id)

    try:
        task = Task.objects.get(id=task_id)
    except Task.DoesNotExist:
        logger.error(f"任务 {task_id} 在开始执行前已不存在，任务终止。")
        return {"status": "aborted", "reason": f"Task {task_id} does not exist."}

    # --- 核心修复：统一使用 pathlib.Path 处理所有路径 ---
    result_dir = Path(settings.MEDIA_ROOT) / 'results' / str(task_id)
    extract_dir = Path(settings.MEDIA_ROOT) / 'extracted' / str(task_id)
    config_path = result_dir / 'config.json'
    result_file_path = result_dir / 'result.csv'

    try:
        task.status = 'processing'
        if task.progress < 1:
            task.progress = 1
        task.celery_task_id = self.request.id
        task.save(update_fields=['status', 'progress', 'celery_task_id'])

        # 进入处理阶段后先给一个可见进度，避免前端长期显示0
        _update_task_progress(task, 2, self)

        update_progress = create_progress_tracker(task, task_id, self)
        
        result_dir.mkdir(exist_ok=True, parents=True) # 确保目录存在
        _update_task_progress(task, 5, self)
        
        # ==================== 图像分类逻辑 ====================
        if task.task_type == 'image-classification':
            # 1. 校验数据文件内容
            valid_image_extensions = {'.jpg', '.jpeg', '.png'}
            image_files = sorted(
                (f for f in extract_dir.rglob('*') if f.suffix.lower() in valid_image_extensions),
                key=lambda p: str(p).lower()
            )
            if not image_files:
                raise ValueError(f"数据压缩包中未找到任何支持的图像文件 ({', '.join(valid_image_extensions)})。")

            # 2. 合并原始标注与人工补标；若没有原始标注，则使用人工补标文件作为训练输入
            manual_label_path = Path(settings.MEDIA_ROOT) / 'results' / str(task.id) / 'manual_labels.csv'
            has_uploaded_label_file = bool(task.label_file and Path(task.label_file.path).exists())
            if not manual_label_path.exists():
                ImageResult = apps.get_model('data_management', 'ImageResult')
                manual_items = ImageResult.objects.filter(task=task, confidence=-1).exclude(label__exact='')
                if manual_items.exists():
                    manual_label_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(manual_label_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['filename', 'class'])
                        for item in manual_items:
                            if item.image_name and item.label:
                                writer.writerow([item.image_name, item.label])
            has_manual_label_file = manual_label_path.exists()
            if not has_uploaded_label_file and not has_manual_label_file:
                def _cold_start_image_uncertainty(image_path):
                    try:
                        from PIL import Image
                        with Image.open(image_path) as img:
                            gray = img.convert('L').resize((64, 64))
                            histogram = gray.histogram()
                            total = float(sum(histogram)) or 1.0
                            entropy = -sum(
                                (count / total) * math.log2(count / total)
                                for count in histogram
                                if count
                            )
                            return round(max(0.0, min(1.0, entropy / 8.0)), 6)
                    except Exception:
                        try:
                            return round(max(0.0, min(1.0, image_path.stat().st_size / (1024 * 1024))), 6)
                        except Exception:
                            return 0.0

                result_rows = []
                for image_path in image_files:
                    result_rows.append({
                        'image_name': image_path.name,
                        'image_path': str(image_path.resolve()),
                        'label': '',
                        'confidence': '',
                        'uncertainty_score': _cold_start_image_uncertainty(image_path),
                    })

                with open(result_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=['image_name', 'image_path', 'label', 'confidence', 'uncertainty_score']
                    )
                    writer.writeheader()
                    writer.writerows(result_rows)

                _update_task_progress(task, 30, self)
                save_results_to_database(result_file_path, task, 'image-classification', update_progress)
                task.progress = 100
                task.status = 'completed'
                task.save()
                logger.info(f"鍗婄洃鐫ｄ换鍔?{task_id} 澶勭悊瀹屾垚")
                return {"status": "success", "task_id": task_id}

            label_file_path = _merge_manual_labels_if_needed(task)
            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    header = f.readline().strip().lower()
                    if 'filename' not in header or 'class' not in header:
                        raise ValueError("标注文件的第一行（表头）必须包含 'filename' 和 'class' 列。")
            except Exception as e:
                raise ValueError(f"读取或校验标注文件失败: {e}")

            _update_task_progress(task, 10, self)

            # 3. 创建配置文件和执行命令
            # 根据前端选择的 model_strength 生成 adapter 配置
            strength_raw = (task.model_strength or "").strip().lower()
            if strength_raw == "high":
                adapter_cfg = {"enabled": True, "num_layers": 2, "hidden_dim": 1024, "activation": "gelu"}
            elif strength_raw == "medium":
                adapter_cfg = {"enabled": True, "num_layers": 1, "hidden_dim": 512, "activation": "relu"}
            else:
                adapter_cfg = {"enabled": False}

            config = {
                "label_csv": str(label_file_path),  # 确保使用具体CSV文件路径
                "dataset_format": "csv",
                "image_dir": str(extract_dir),  # 将 Path 对象转为字符串
                "embedding_save_path": str(result_dir / "embedding/"), # 统一使用 Path 和 str()
                "output_path": str(result_dir),  # 统一使用 Path 和 str()
                "batch_size": 64,
                "device": "cuda",
                # CG3 / diffmap / manifold 超参
                "cg3_k": 30,
                "cg3_lr": 5e-4,
                "cg3_epochs": 2000,
                "cg3_threshold": 0.9,
                "cg3_num_layers": 5,
                "cg3_hidden_dim": 512,
                "cg3_out_dim": 256,
                "diffmap_k": 30,
                "diffmap_sigma": 0.3,
                "diffmap_components": 512,
                "diffmap_epochs": 60,
                "diffmap_lr": 0.003,
                "llgc_alpha": 0.6,
                "manifold_lambda": 0.2,
                "manifold_lr": 0.003,
                "manifold_epochs": 60,
                "manifold_num_layers": 3,
                "manifold_hidden_dims": [512, 256, 128],
                # 将 adapter 配置传递给 ImageCLIPEncoder
                "adapter": adapter_cfg,
                # 默认为 ViT-B/32，可通过前端或策略改为其他变体
                "clip_model": "ViT-B/32",
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)

            _update_task_progress(task, 15, self)
            
            algorithm_script_path = Path(settings.BASE_DIR) / 'smartlabel' / 'apps' / 'algorithm' / 'img_classfication' / 'train.py'
            cmd = ['conda', 'run', '--no-capture-output', '-n', 'img_classification', 'python', str(algorithm_script_path), '--config', str(config_path)]

            # 算法主流程开始，进度先推进到20
            _update_task_progress(task, 20, self)
            run_subprocess_with_progress(cmd, task_id)

            # 子进程结束，推进到80，后续由结果落库推进到100
            _update_task_progress(task, 80, self)

            if not result_file_path.exists():
                raise FileNotFoundError(f"图像分类结果文件不存在: {result_file_path}")
            _update_task_progress(task, 85, self)
            save_results_to_database(result_file_path, task, 'image-classification', update_progress)

        # ==================== 文本分类逻辑 ====================
        elif task.task_type == 'text-classification':
            unlabeled_files = list(extract_dir.glob('*.csv'))
            if not unlabeled_files:
                raise FileNotFoundError(f"数据压缩包解压后，在目录 {extract_dir} 中未找到任何 .csv 文件。")

            TEXT_COLUMN_KEYWORDS = ['text', 'content', 'body', 'sentence', 'article', 'description', 'message']
            manual_text_label_path = _manual_text_label_path(task)
            if not manual_text_label_path.exists():
                TextResult = apps.get_model('data_management', 'TextResult')
                manual_items = TextResult.objects.filter(task=task, confidence=-1).exclude(label__exact='')
                if manual_items.exists():
                    manual_text_label_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(manual_text_label_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['text', 'label'])
                        for item in manual_items:
                            if item.content and item.label:
                                writer.writerow([item.content, item.label])
            has_text_label_file = bool(task.label_file and Path(task.label_file.path).exists())
            has_manual_text_label_file = manual_text_label_path.exists()

            if not has_text_label_file and not has_manual_text_label_file:
                # 无标签文本任务：只输出不确定性，预测标签和置信度保持空。
                texts = []
                for unlabeled_file in unlabeled_files:
                    try:
                        df = pd.read_csv(unlabeled_file, encoding='utf-8')
                        text_col = next((col for col in df.columns if col.strip().lower() in TEXT_COLUMN_KEYWORDS), df.columns[0] if len(df.columns) > 0 else None)
                        if text_col:
                            texts.extend(df[text_col].fillna('').astype(str).tolist())
                    except Exception as e:
                        logger.warning(f"读取无标签文件 {unlabeled_file.name} 失败: {e}。已跳过。")

                if not texts:
                    raise ValueError("未能从数据文件中成功提取任何无标签文本。")

                lengths = [max(len(text.split()), 1) for text in texts]
                min_len = min(lengths)
                max_len = max(lengths)
                length_span = max(max_len - min_len, 1)

                result_rows = []
                for index, text in enumerate(texts, start=1):
                    length_score = max(0.0, min(1.0, 1.0 - ((max(len(text.split()), 1) - min_len) / length_span)))
                    uncertainty_score = round(float(length_score), 6)
                    result_rows.append({
                        'text_id': str(index),
                        'content': text,
                        'label': '',
                        'confidence': '',
                        'uncertainty_score': uncertainty_score,
                    })

                result_dir.mkdir(parents=True, exist_ok=True)
                with open(result_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['text_id', 'content', 'label', 'confidence', 'uncertainty_score'])
                    writer.writeheader()
                    writer.writerows(result_rows)

                _update_task_progress(task, 30, self)
                save_results_to_database(result_file_path, task, 'text-classification', update_progress)

            else:
                label_file_path = Path(task.label_file.path) if has_text_label_file else manual_text_label_path
                try:
                    with open(label_file_path, 'r', encoding='utf-8') as f:
                        header = f.readline().strip().lower()
                        if 'text' not in header or 'label' not in header:
                            raise ValueError("标注文件的第一行（表头）必须包含 'text' 和 'label' 列。")

                    labeled_df = pd.read_csv(label_file_path, encoding='utf-8')
                    text_col_labeled = next((col for col in labeled_df.columns if 'text' in col.lower()), 'text')
                    label_col_labeled = next((col for col in labeled_df.columns if 'label' in col.lower() or 'class' in col.lower()), 'label')
                    labeled_df = labeled_df[[text_col_labeled, label_col_labeled]].rename(columns={text_col_labeled: 'text', label_col_labeled: 'label'})
                    labeled_df['is_labeled'] = True
                    logger.info(f"成功从 {label_file_path.name} 加载 {len(labeled_df)} 条有标签数据。")
                except Exception as e:
                    raise ValueError(f"读取或处理标注文件失败: {e}")

                _update_task_progress(task, 10, self)

                unlabeled_data_list = []
                for unlabeled_file in unlabeled_files:
                    try:
                        df = pd.read_csv(unlabeled_file, encoding='utf-8')
                        text_col_unlabeled = next((col for col in df.columns if col.strip().lower() in TEXT_COLUMN_KEYWORDS), df.columns[0] if len(df.columns) > 0 else None)
                        if text_col_unlabeled:
                            temp_df = df[[text_col_unlabeled]].rename(columns={text_col_unlabeled: 'text'}).copy()
                            temp_df['label'] = ''
                            unlabeled_data_list.append(temp_df)
                    except Exception as e:
                        logger.warning(f"读取无标签文件 {unlabeled_file.name} 失败: {e}。已跳过。")

                if not unlabeled_data_list:
                    raise ValueError("未能从数据文件中成功提取任何无标签文本。")
                unlabeled_df = pd.concat(unlabeled_data_list, ignore_index=True)
                unlabeled_df['is_labeled'] = False
                logger.info(f"成功加载 {len(unlabeled_df)} 条无标签数据。")

                _update_task_progress(task, 15, self)

                combined_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
                combined_df['text'] = combined_df['text'].astype(str)
                combined_df['is_labeled'] = combined_df['is_labeled'].astype(bool)
                combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')

                combined_data_file = result_dir / 'combined_dataset.csv'
                combined_df.to_csv(combined_data_file, index=False, encoding='utf-8')
                logger.info(f"合并后的数据集已保存到: {combined_data_file}")

                _update_task_progress(task, 20, self)

                result_dir.mkdir(parents=True, exist_ok=True)
                embedding_dir = result_dir / "embeddings"
                embedding_dir.mkdir(parents=True, exist_ok=True)

                # 主动学习冷启动要求：至少要有一个人工标注样本，否则模型无法形成有效类别覆盖
                labeled_label_count = int(combined_df.loc[combined_df['is_labeled'] == True, 'label'].nunique())
                if labeled_label_count == 0:
                    raise ValueError("文本分类任务至少需要 1 个已人工标注类别样本，才能启动训练与主动学习流程。")

                # 所有文本分类任务现在统一使用 CG3 闭集分类算法（原主题分类逻辑）
                logger.info("文本分类任务，调用 flexmatch CG3 基础文本分类算法")
                
                config = {
                    "dataset_path": str(combined_data_file),
                    "dataset_format": "csv",
                    "classification_scene": "topic",
                    "embedding_save_path": str(embedding_dir),
                    "output_path": str(result_dir),
                    "language": "zh",
                    "bert_model": _resolve_text_classification_bert_model(),
                    "batch_size": 32,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "cg3_use": True,
                    "cg3_epochs": 200,
                    "cg3_lr": 0.001,
                    "cg3_k": 15,
                    "cg3_threshold": 0.7,
                    "cg3_sigma": 0.5,
                    "cg3_warmup": 10,
                    "llgc_use": False,
                    "manifold_use": False,
                    "mixtext_use": False,
                }
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)

                logger.info(f"配置文件已保存到: {config_path}")
                _update_task_progress(task, 25, self)

                # 调用 flexmatch 文本分类脚本
                algorithm_script_path = Path(settings.BASE_DIR) / 'smartlabel' / 'apps' / 'algorithm' / 'flexmatch' / 'train.py'
                cmd = ['conda', 'run', '--no-capture-output', '-n', 'text_classification', 'python', str(algorithm_script_path), '--config', str(config_path)]

                logger.info(f"执行命令: {' '.join(shlex.quote(s) for s in cmd)}")
                _update_task_progress(task, 30, self)
                run_subprocess_with_progress(cmd, task_id)
                _update_task_progress(task, 80, self)

                task.refresh_from_db()
                update_progress(98, 100)

                if not result_file_path.exists():
                    raise FileNotFoundError(f"文本分类结果文件不存在: {result_file_path}")
                _update_task_progress(task, 85, self)
                save_results_to_database(result_file_path, task, 'text-classification', update_progress)

        # ==================== 目标检测逻辑 ====================
        elif task.task_type == 'object-detection':
            # 验证数据目录
            valid_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
            image_files = [p for p in extract_dir.rglob('*') if p.suffix.lower() in valid_image_extensions]
            if not image_files:
                raise FileNotFoundError(f"数据压缩包中未找到任何支持的图像文件 ({', '.join(valid_image_extensions)})。")

            _update_task_progress(task, 10, self)

            # 从数据库中读取人工标注（前端保存到 ImageResult.annotations）作为有标签样本
            ImageResult = apps.get_model('data_management', 'ImageResult')
            labeled_map = {}
            external_label_loaded = False

            image_by_name = {p.name: p for p in image_files}

            def _canonical_detection_label(label):
                label_text = str(label or '').strip()
                if not label_text:
                    return 'obj'
                return DETECTION_LABEL_ALIASES.get(label_text, label_text)

            def _append_coco_annotations(coco_data):
                categories = {
                    int(c.get('id')): _canonical_detection_label(c.get('name') or 'obj')
                    for c in coco_data.get('categories', [])
                    if c.get('id') is not None
                }
                images_meta = {
                    int(img.get('id')): img
                    for img in coco_data.get('images', [])
                    if img.get('id') is not None
                }

                for ann in coco_data.get('annotations', []):
                    image_id = ann.get('image_id')
                    bbox = ann.get('bbox') or []
                    category_id = ann.get('category_id')
                    if image_id is None or len(bbox) < 4:
                        continue

                    image_meta = images_meta.get(int(image_id))
                    if not image_meta:
                        continue

                    file_name = Path(str(image_meta.get('file_name', ''))).name
                    if not file_name:
                        continue

                    img_path = image_by_name.get(file_name)
                    if not img_path:
                        continue

                    img_w = float(image_meta.get('width') or 0)
                    img_h = float(image_meta.get('height') or 0)
                    if img_w <= 0 or img_h <= 0:
                        continue

                    x, y, w, h = [float(v) for v in bbox[:4]]
                    if w <= 0 or h <= 0:
                        continue

                    xc = (x + w / 2.0) / img_w
                    yc = (y + h / 2.0) / img_h
                    wn = w / img_w
                    hn = h / img_h

                    lbl = categories.get(int(category_id), 'obj') if category_id is not None else 'obj'
                    labeled_map.setdefault(img_path.name, []).append({
                        'label': lbl,
                        'x': max(0.0, min(1.0, xc)),
                        'y': max(0.0, min(1.0, yc)),
                        'w': max(0.0, min(1.0, wn)),
                        'h': max(0.0, min(1.0, hn)),
                    })

            # 若上传了标注文件（可选），优先尝试解析 JSON(COCO) 或 CSV
            if task.label_file and Path(task.label_file.path).exists():
                label_file_path = Path(task.label_file.path)
                try:
                    if label_file_path.suffix.lower() == '.json':
                        with open(label_file_path, 'r', encoding='utf-8') as jf:
                            label_json = json.load(jf)
                        if isinstance(label_json, dict) and 'images' in label_json and 'annotations' in label_json:
                            before_label_count = len(labeled_map)
                            _append_coco_annotations(label_json)
                            external_label_loaded = external_label_loaded or len(labeled_map) > before_label_count
                            logger.info("任务 %s 从上传标注 JSON %s 解析到 %s 张有标注图片", task_id, label_file_path.name, len(labeled_map))
                    elif label_file_path.suffix.lower() == '.csv':
                        with open(label_file_path, 'r', encoding='utf-8', newline='') as cf:
                            reader = csv.DictReader(cf)
                            for row in reader:
                                fname = (row.get('filename') or row.get('image') or row.get('image_name') or '').strip()
                                if not fname:
                                    continue
                                img_name = Path(fname).name
                                img_path = image_by_name.get(img_name)
                                if not img_path:
                                    continue
                                try:
                                    xmin = float(row.get('xmin', ''))
                                    ymin = float(row.get('ymin', ''))
                                    xmax = float(row.get('xmax', ''))
                                    ymax = float(row.get('ymax', ''))
                                except Exception:
                                    continue
                                if xmax <= xmin or ymax <= ymin:
                                    continue

                                # 使用图片实际尺寸做归一化，避免依赖 CSV 附加宽高字段
                                from PIL import Image
                                with Image.open(img_path) as im:
                                    iw, ih = im.size
                                if iw <= 0 or ih <= 0:
                                    continue
                                x = (xmin + xmax) / 2.0 / iw
                                y = (ymin + ymax) / 2.0 / ih
                                w = (xmax - xmin) / iw
                                h = (ymax - ymin) / ih
                                lbl = _canonical_detection_label((row.get('class') or row.get('label') or 'obj').strip() or 'obj')

                                labeled_map.setdefault(img_name, []).append({
                                    'label': lbl,
                                    'x': max(0.0, min(1.0, x)),
                                    'y': max(0.0, min(1.0, y)),
                                    'w': max(0.0, min(1.0, w)),
                                    'h': max(0.0, min(1.0, h)),
                                })
                        external_label_loaded = True
                        logger.info("任务 %s 从上传标注 CSV %s 解析到 %s 张有标注图片", task_id, label_file_path.name, len(labeled_map))
                except Exception as e:
                    logger.warning("任务 %s 解析上传标注文件失败: %s", task_id, e)

            # 若数据库中暂无人工标注，尝试解析上传包中的 COCO 标注（annotations/instances*.json）。
            if not external_label_loaded:
                try:
                    coco_candidates = sorted(
                        list(Path(extract_dir).rglob('instances*.json'))
                        + list(Path(extract_dir).rglob('annotations*.json'))
                    )
                    for coco_file in coco_candidates:
                        with open(coco_file, 'r', encoding='utf-8') as cf:
                            coco_data = json.load(cf)

                        before_label_count = len(labeled_map)
                        _append_coco_annotations(coco_data)
                        external_label_loaded = external_label_loaded or len(labeled_map) > before_label_count

                        if labeled_map:
                            logger.info("任务 %s 从 COCO 标注文件 %s 解析到 %s 张有标注图片", task_id, coco_file, len(labeled_map))
                            break
                except Exception as e:
                    logger.warning("任务 %s 解析 COCO 标注失败，将继续使用人工标注流程: %s", task_id, e)

            db_manual_map = {}
            for ir in ImageResult.objects.filter(task=task).all():
                if not ir.annotations:
                    continue
                confirmed_annotations = []
                for annotation in ir.annotations:
                    if not isinstance(annotation, dict):
                        continue
                    source = annotation.get('source') or ''
                    decision = annotation.get('decision') or annotation.get('annotation_status') or ''
                    if ir.confidence == -1 or source == 'manual' or decision in ('accepted', 'manual'):
                        confirmed_annotations.append(annotation)
                if confirmed_annotations:
                    db_manual_map[ir.image_name] = confirmed_annotations
            labeled_map.update(db_manual_map)
            if db_manual_map:
                logger.info(
                    "任务 %s 合并数据库人工确认标注 %s 张图片；训练标注总计 %s 张图片",
                    task_id,
                    len(db_manual_map),
                    len(labeled_map),
                )

            def _annotation_to_yolo_box(annotation):
                """将界面标注规范化为 YOLO 训练需要的中心点框。

                目标检测训练只吃矩形框；界面允许画 polygon 时，取外接矩形作为训练框。
                """
                if not isinstance(annotation, dict):
                    return None

                source = annotation.get('source') or (
                    'model' if annotation.get('confidence') not in (None, '') else 'manual'
                )
                decision = annotation.get('decision') or annotation.get('annotation_status') or (
                    'pending' if source == 'model' else 'manual'
                )
                if source == 'model' and decision in ('pending', 'rejected'):
                    return None
                if decision == 'rejected':
                    return None

                label = _canonical_detection_label(annotation.get('label') or '')
                shape_type = annotation.get('shape_type') or 'rectangle'

                if shape_type == 'rectangle':
                    points = annotation.get('points') or []
                    if isinstance(points, list) and len(points) >= 2:
                        try:
                            xs = [max(0.0, min(1.0, float(p[0]))) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
                            ys = [max(0.0, min(1.0, float(p[1]))) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
                            if len(xs) >= 2 and len(ys) >= 2:
                                min_x, max_x = min(xs), max(xs)
                                min_y, max_y = min(ys), max(ys)
                                w = max_x - min_x
                                h = max_y - min_y
                                if w > 0 and h > 0:
                                    return {
                                        'label': label,
                                        'x': min_x + w / 2.0,
                                        'y': min_y + h / 2.0,
                                        'w': w,
                                        'h': h,
                                    }
                        except Exception:
                            pass

                    try:
                        x = float(annotation.get('x', 0))
                        y = float(annotation.get('y', 0))
                        w = float(annotation.get('w', 0))
                        h = float(annotation.get('h', 0))
                    except (TypeError, ValueError):
                        return None
                    if w <= 0 or h <= 0:
                        return None
                    return {
                        'label': label,
                        'x': max(0.0, min(1.0, x)),
                        'y': max(0.0, min(1.0, y)),
                        'w': max(0.0, min(1.0, w)),
                        'h': max(0.0, min(1.0, h)),
                    }

                if shape_type == 'polygon':
                    points = annotation.get('points') or []
                    try:
                        xs = [max(0.0, min(1.0, float(p[0]))) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
                        ys = [max(0.0, min(1.0, float(p[1]))) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
                    except (TypeError, ValueError):
                        return None
                    if len(xs) < 3 or len(ys) < 3:
                        return None
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    w = max_x - min_x
                    h = max_y - min_y
                    if w <= 0 or h <= 0:
                        return None
                    return {
                        'label': label,
                        'x': min_x + w / 2.0,
                        'y': min_y + h / 2.0,
                        'w': w,
                        'h': h,
                    }

                return None

            # 收集类别名称时只统计真正能进入训练的标注，避免“有 annotations 但无可训练框”的假阳性。
            class_names = set()
            usable_labeled_map = {}
            for image_name, anns in labeled_map.items():
                usable_boxes = []
                for ann in anns:
                    box = _annotation_to_yolo_box(ann)
                    if not box:
                        continue
                    box['label'] = _canonical_detection_label(box.get('label'))
                    if box.get('label'):
                        class_names.add(box['label'])
                    usable_boxes.append(box)
                if usable_boxes:
                    usable_labeled_map[image_name] = usable_boxes

            labeled_map = usable_labeled_map

            # 如果 Task 中预设了 label_list，优先使用
            if task.label_list:
                class_list = []
                for label_name in task.label_list:
                    canonical_label = _canonical_detection_label(label_name)
                    if canonical_label not in class_list:
                        class_list.append(canonical_label)
            else:
                class_list = sorted(list(class_names))

            if not class_list:
                class_list = ['obj']

            class2id = {n: i for i, n in enumerate(class_list)}

            def _label_path_for_image(img_path):
                parts = list(img_path.parts)
                if 'images' in parts:
                    idx = parts.index('images')
                    parts[idx] = 'labels'
                    return Path(*parts).with_suffix('.txt')
                return img_path.parent / 'labels' / f"{img_path.stem}.txt"

            # 为每张图片生成 YOLO 格式的 label 文件（放在解压目录旁，跟图片同目录）
            for img_path in image_files:
                stem = img_path.stem
                anns = labeled_map.get(img_path.name) or []
                label_lines = []
                for a in anns:
                    try:
                        x = float(a.get('x', 0))
                        y = float(a.get('y', 0))
                        w = float(a.get('w', 0))
                        h = float(a.get('h', 0))
                        if w <= 0 or h <= 0:
                            continue
                        lbl = a.get('label') or class_list[0]
                        cid = class2id.get(lbl, 0)
                        label_lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    except Exception:
                        continue

                label_file = _label_path_for_image(img_path)
                label_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(label_file, 'w', encoding='utf-8') as lf:
                        for ln in label_lines:
                            lf.write(ln + '\n')
                except Exception as e:
                    logger.warning(f"无法写入标签文件 {label_file}: {e}")

            _update_task_progress(task, 20, self)

            # 生成 train/val/target 列表
            labeled_images = [
                p for p in image_files
                if _label_path_for_image(p).exists() and _label_path_for_image(p).stat().st_size > 0
            ]
            unlabeled_images = [p for p in image_files if p not in labeled_images]

            if not labeled_images:
                raise ValueError("目标检测任务未检测到可用标注。请上传含 COCO/VOC/YOLO 标注的数据压缩包，或先在界面完成至少1张人工标注后再提交。")

            # 简单划分 val（10% 的有标签样本）
            if len(labeled_images) == 1:
                train_images = labeled_images
                val_images = labeled_images
            else:
                val_count = max(1, int(len(labeled_images) * 0.1))
                if val_count >= len(labeled_images):
                    val_count = len(labeled_images) - 1
                val_images = labeled_images[:val_count]
                train_images = labeled_images[val_count:]

            # EfficientTeacher 某些流程会读取 target 列表，避免空文件触发上游 list index 错误。
            if not unlabeled_images:
                unlabeled_images = train_images

            data_list_dir = Path(result_dir) / 'data_lists'
            data_list_dir.mkdir(parents=True, exist_ok=True)

            train_txt = data_list_dir / 'train.txt'
            val_txt = data_list_dir / 'val.txt'
            target_txt = data_list_dir / 'target.txt'

            def write_list(path, lst, with_label=False):
                with open(path, 'w', encoding='utf-8') as f:
                    for p in lst:
                        if with_label:
                            f.write(f"{str(p.resolve())} {str(_label_path_for_image(p).resolve())}\n")
                        else:
                            f.write(str(p.resolve()) + '\n')

            write_list(train_txt, train_images, with_label=True)
            write_list(val_txt, val_images, with_label=True)
            write_list(target_txt, unlabeled_images)

            # EfficientTeacher 会优先读取 data list 同名的 .cache；旧缓存不会总是校验 hash，
            # 重新人工标注后如果不清理，会继续使用旧标签并导致类别越界等训练错误。
            for list_path in (train_txt, val_txt, target_txt):
                cache_path = list_path.with_suffix('.cache')
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        logger.info("已删除旧数据集缓存: %s", cache_path)
                    except Exception as e:
                        logger.warning("删除旧数据集缓存失败 %s: %s", cache_path, e)

            _update_task_progress(task, 30, self)

            # 创建 EfficientTeacher 配置文件（基于官方 yolov5 模板，避免默认值触发 NotImplementedError）
            cfg_name = f"task_{task_id}_effteacher"
            cfg_path = Path(result_dir) / 'config.yaml'
            cfg_lines = []
            cfg_lines.append(f"project: '{result_dir}'")
            cfg_lines.append(f"name: '{cfg_name}'")
            cfg_lines.append("epochs: 5")
            cfg_lines.append("adam: False")
            pretrained_weight = str(DETECTION_PRETRAINED_WEIGHT) if DETECTION_PRETRAINED_WEIGHT.exists() else ''
            cfg_lines.append(f"weights: '{pretrained_weight}'")
            if pretrained_weight:
                logger.info("目标检测训练使用预训练权重: %s", pretrained_weight)
            else:
                logger.warning(
                    "未找到目标检测预训练权重 %s，将从零训练；少量数据下可能无法生成可靠预测框。",
                    DETECTION_PRETRAINED_WEIGHT,
                )
            cfg_lines.append("prune_finetune: False")

            cfg_lines.append("hyp:")
            cfg_lines.append("  lr0: 0.01")
            cfg_lines.append("  hsv_h: 0.015")
            cfg_lines.append("  hsv_s: 0.7")
            cfg_lines.append("  hsv_v: 0.4")
            cfg_lines.append("  lrf: 0.01")

            cfg_lines.append("Model:")
            cfg_lines.append("  depth_multiple: 0.33")
            cfg_lines.append("  width_multiple: 0.50")
            cfg_lines.append("  Backbone:")
            cfg_lines.append("    name: 'YoloV5'")
            cfg_lines.append("    activation: 'SiLU'")
            cfg_lines.append("  Neck:")
            cfg_lines.append("    name: 'YoloV5'")
            cfg_lines.append("    in_channels: [256, 512, 1024]")
            cfg_lines.append("    out_channels: [256, 512, 1024]")
            cfg_lines.append("    activation: 'SiLU'")
            cfg_lines.append("  Head:")
            cfg_lines.append("    name: 'YoloV5'")
            cfg_lines.append("    activation: 'SiLU'")
            cfg_lines.append("  anchors: [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]")

            cfg_lines.append("Loss:")
            cfg_lines.append("  type: 'ComputeLoss'")

            cfg_lines.append("Dataset:")
            cfg_lines.append(f"  train: {str(train_txt)}")
            cfg_lines.append(f"  val: {str(val_txt)}")
            cfg_lines.append(f"  target: {str(target_txt)}")
            cfg_lines.append("  data_name: 'custom'")
            cfg_lines.append(f"  nc: {len(class_list)}")
            cfg_lines.append(f"  img_size: 640")
            cfg_lines.append(f"  batch_size: 16")
            # names 列表展开为行
            cfg_lines.append("  names: [" + ", ".join([f'\'{n}\'' for n in class_list]) + "]")

            with open(cfg_path, 'w', encoding='utf-8') as cf:
                cf.write('\n'.join(cfg_lines))

            _update_task_progress(task, 35, self)

            # 调用 EfficientTeacher 的 train.py
            efficient_train = Path(settings.BASE_DIR) / 'smartlabel' / 'apps' / 'algorithm' / 'efficientteacher-main' / 'train.py'
            # 使用 conda 环境名 efficientteacher；如无请用户自行准备
            cmd = [
                'conda', 'run', '--no-capture-output', '-n', 'efficientteacher', 'python',
                str(efficient_train), '--cfg', str(cfg_path)
            ]

            logger.info(f"执行目标检测训练命令: {' '.join(shlex.quote(s) for s in cmd)}")
            _update_task_progress(task, 40, self)
            run_subprocess_with_progress(cmd, task_id)

            _update_task_progress(task, 75, self)

            # 训练完成后，用 detect.py 在所有图片上做预测并收集结果
            detect_py = Path(settings.BASE_DIR) / 'smartlabel' / 'apps' / 'algorithm' / 'efficientteacher-main' / 'detect.py'
            # EfficientTeacher 会在同名目录已存在时自动追加编号，例如 task_1_effteacher2。
            # 因此必须按最新训练目录寻找权重，不能只查固定目录。
            weight_candidates = []
            for run_dir in sorted(Path(result_dir).glob(f'{cfg_name}*'), key=lambda p: p.stat().st_mtime, reverse=True):
                for candidate_name in ('best.pt', 'last.pt'):
                    candidate = run_dir / 'weights' / candidate_name
                    if candidate.exists():
                        weight_candidates.append(candidate)
            if not weight_candidates:
                raise FileNotFoundError(f"目标检测训练已结束，但未找到输出权重: {Path(result_dir) / (cfg_name + '*') / 'weights'}")
            weight_file = weight_candidates[0]
            logger.info("目标检测推理使用权重: %s", weight_file)

            detect_source = extract_dir / 'images' if (extract_dir / 'images').exists() else extract_dir
            detect_cmd = [
                'conda', 'run', '--no-capture-output', '-n', 'efficientteacher', 'python',
                str(detect_py),
                '--weights', str(weight_file),
                '--source', str(detect_source),
                '--imgsz', '640',
                '--conf-thres', str(DETECTION_PREDICT_CONF_THRES),
                '--iou-thres', str(DETECTION_PREDICT_IOU_THRES),
                '--max-det', str(DETECTION_PREDICT_MAX_DET),
                '--save-txt',
                '--save-conf',
                '--project', str(result_dir),
                '--name', 'pred',
                '--exist-ok',
            ]
            logger.info(
                "object detection predict args: source=%s, conf_thres=%s, iou_thres=%s, max_det=%s",
                detect_source,
                DETECTION_PREDICT_CONF_THRES,
                DETECTION_PREDICT_IOU_THRES,
                DETECTION_PREDICT_MAX_DET,
            )
            pred_dir = Path(result_dir) / 'pred'
            if pred_dir.exists():
                import shutil
                shutil.rmtree(pred_dir)
            run_subprocess_with_progress(detect_cmd, task_id)

            # 解析 detect 输出 labels，生成 result.csv 并落库
            pred_labels_dir = pred_dir / 'labels'
            if not pred_labels_dir.exists():
                raise FileNotFoundError(f"目标检测推理未生成标签目录: {pred_labels_dir}")

            results_rows = []
            seq = 0

            for img_path in image_files:
                seq += 1
                img_name = img_path.name
                rel_path = str(img_path.resolve())
                label_file = pred_labels_dir / (img_path.stem + '.txt')
                annotations = []
                max_conf = 0.0
                if label_file.exists():
                    with open(label_file, 'r', encoding='utf-8') as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            if len(parts) >= 5:
                                cid = int(float(parts[0]))
                                x = float(parts[1])
                                y = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                conf = float(parts[5]) if len(parts) >= 6 else 0.0
                                annotations.append({
                                    'label': class_list[cid] if cid < len(class_list) else str(cid),
                                    'x': x,
                                    'y': y,
                                    'w': w,
                                    'h': h,
                                    'confidence': conf,
                                    'shape_type': 'rectangle',
                                    'source': 'model',
                                    'decision': 'pending',
                                    'sample_mode': 'model_prediction',
                                })
                    annotations.sort(key=lambda item: float(item.get('confidence') or 0.0), reverse=True)
                    annotations = annotations[:DETECTION_MAX_BOXES_PER_IMAGE]
                    max_conf = max((float(item.get('confidence') or 0.0) for item in annotations), default=0.0)

                uncertainty_score, low_confidence_count = detection_uncertainty(annotations)
                results_rows.append({
                    'sequence_number': seq,
                    'image_name': img_name,
                    'image_path': rel_path,
                    'annotations': annotations,
                    'confidence': max_conf,
                    'uncertainty_score': uncertainty_score,
                    'low_confidence_count': low_confidence_count,
                    'detection_count': len(annotations),
                    'status': 'unverified'
                })

            # 将结果写入数据库（ImageResult）
            predicted_image_count = sum(1 for row in results_rows if row['annotations'])
            predicted_box_count = sum(len(row['annotations']) for row in results_rows)
            logger.info(
                "object detection predict finished: %s/%s images with boxes, %s boxes total",
                predicted_image_count,
                len(results_rows),
                predicted_box_count,
            )
            if predicted_box_count == 0:
                logger.warning(
                    "目标检测推理没有生成任何高置信候选框；将保留未标注图片为空框状态，避免写入低置信度噪声框。"
                )

            with open(result_file_path, 'w', encoding='utf-8', newline='') as rf:
                writer = csv.DictWriter(
                    rf,
                    fieldnames=[
                        'sequence_number',
                        'image_name',
                        'image_path',
                        'annotations',
                        'confidence',
                        'uncertainty_score',
                        'low_confidence_count',
                        'detection_count',
                        'status',
                    ],
                )
                writer.writeheader()
                for row in results_rows:
                    csv_row = row.copy()
                    csv_row['annotations'] = json.dumps(csv_row['annotations'], ensure_ascii=False)
                    writer.writerow(csv_row)

            ResultModel = apps.get_model('data_management', 'ImageResult')

            manual_image_names = set(
                ResultModel.objects
                .filter(task=task, confidence=-1)
                .values_list('image_name', flat=True)
            )

            with transaction.atomic():
                # 只清理模型预测结果，保留人工标注 ground truth。
                ResultModel.objects.filter(task=task).exclude(confidence=-1).delete()
                instances = []
                for r in results_rows:
                    if r['image_name'] in manual_image_names:
                        continue
                    instances.append(ResultModel(
                        task=task,
                        sequence_number=r['sequence_number'],
                        image_name=r['image_name'],
                        image_path=r['image_path'],
                        label='',
                        confidence=r['confidence'],
                        annotations=r['annotations'],
                        status='unverified'
                    ))
                if instances:
                    ResultModel.objects.bulk_create(instances)

            _update_task_progress(task, 90, self)
            # 最终推进到完成状态由外层逻辑处理

        # --- 任务完成 ---
        task.progress = 100
        task.status = 'completed'
        task.save()
        
        logger.info(f"半监督任务 {task_id} 处理完成")
        return {"status": "success", "task_id": task_id}
        
    except (FileNotFoundError, ValueError) as e:
         logger.error(f"处理半监督任务 {task_id} 时发生数据校验错误: {str(e)}")
         task.status = 'failed'
         task.error_log = f"数据校验失败: {e}"
         task.save(update_fields=['status', 'error_log'])
         return {"status": "failed", "reason": str(e)}

    except Exception as e:
        logger.error(f"处理半监督任务 {task_id} 时发生未知错误: {e}", exc_info=True)
        task.status = 'failed'
        task.error_log = f"内部服务器错误: {e}"
        task.save(update_fields=['status', 'error_log'])
        # 不再自动重试，避免训练脚本持续失败时反复占用 worker。
        raise Ignore()
