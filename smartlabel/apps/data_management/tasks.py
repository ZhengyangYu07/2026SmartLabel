import csv
import glob
import json
import logging
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
from .models import Task

MIN_UPDATE_INTERVAL = 0.5
logger = logging.getLogger(__name__)
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


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    max_retries=2,
    retry_backoff=120,
    retry_jitter=True,
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
        
        # 重试，最多重试3次（针对非 Task.DoesNotExist 错误）
        raise self.retry(exc=e, countdown=120)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    max_retries=2,
    retry_backoff=120,
    retry_jitter=True,
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
            if not any(f.suffix.lower() in valid_image_extensions for f in extract_dir.rglob('*')):
                raise ValueError(f"数据压缩包中未找到任何支持的图像文件 ({', '.join(valid_image_extensions)})。")

            # 2. 合并原始标注与人工补标；若没有原始标注，则使用人工补标文件作为训练输入
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

            if not task.label_file or not Path(task.label_file.path).exists():
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
                label_file_path = Path(task.label_file.path)
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
        raise self.retry(exc=e, countdown=120)