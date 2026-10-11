"""
核心任务流水线：Celery 异步执行标注任务。

主入口 process_annotation_task 按 task_type 分三条支线：
- image-classification：调用 img_classification 环境脚本做半监督图像分类；
- text-classification ：调用 text_classification 环境脚本做文本分类；
- object-detection    ：调用 efficientteacher 环境脚本做目标检测的冷启动/主动学习/微调。

三条支线共用的机制：
- 通过 conda 子进程调用各算法环境，run_subprocess_with_progress 实时解析输出推进进度；
- 统一落库到 ImageResult/TextResult，并把检测结果写入闭环表（正样本索引/复核队列）；
- 全程用 Redis 心跳标记算法存活，异常时把任务置为 failed 并记录 error_log。

注意：本文件大量依赖宿主机上的 conda 环境名与模型权重，部署环境需保持一致。
"""
import csv
import json
import logging
import math
import os
import pandas as pd
import psutil
import queue
import shlex
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from celery import shared_task
from celery.exceptions import Ignore
from django.apps import apps
from django.core.cache import cache
from django.conf import settings
from django.db import transaction
from pathlib import Path
import torch
from PIL import Image
from smartlabel.apps.algorithm.object_detection.active_learning import (
    build_detection_review_context,
    detection_review_value_detail,
)
from .models import Task
from .services.detection_loop import (
    get_next_queue_version,
    merge_positive_index_context,
    persist_review_queue_items,
)

# 进度写库的最小间隔（秒），避免子进程高频输出导致数据库被写爆
MIN_UPDATE_INTERVAL = 0.5
logger = logging.getLogger(__name__)
# 目标检测训练/推理参数，均可用环境变量覆盖，便于不同机器调参而无需改代码
DETECTION_TRAIN_EPOCHS = int(os.environ.get('SMARTLABEL_DETECTION_EPOCHS', '50'))
DETECTION_TRAIN_BATCH_SIZE = int(os.environ.get('SMARTLABEL_DETECTION_BATCH_SIZE', '8'))
DETECTION_FREEZE_LAYER_NUM = int(os.environ.get('SMARTLABEL_DETECTION_FREEZE_LAYER_NUM', '10'))
DETECTION_MIN_TRAIN_IMAGES = int(os.environ.get('SMARTLABEL_DETECTION_MIN_TRAIN_IMAGES', '20'))
DETECTION_PREDICT_CONF_THRES = float(os.environ.get('SMARTLABEL_DETECTION_CONF_THRES', '0.05'))
# 推理时低于该置信度的框直接丢弃，不写入结果（避免噪声框）
DETECTION_STORE_CONF_THRES = float(os.environ.get('SMARTLABEL_DETECTION_STORE_CONF_THRES', '0.25'))
DETECTION_PREDICT_IOU_THRES = 0.45
DETECTION_PREDICT_MAX_DET = 100
DETECTION_MAX_BOXES_PER_IMAGE = 100
# 微调结果“过弱”的判定阈值：低于此值则改用基础预训练模型兜底
DETECTION_MIN_FINE_TUNED_BOXES = int(os.environ.get('SMARTLABEL_DETECTION_MIN_FINE_TUNED_BOXES', '50'))
DETECTION_MIN_FINE_TUNED_IMAGE_RATIO = float(os.environ.get('SMARTLABEL_DETECTION_MIN_FINE_TUNED_IMAGE_RATIO', '0.05'))
# 基础预训练权重候选路径，按顺序取第一个存在的
DETECTION_PRETRAINED_WEIGHT_CANDIDATES = [
    Path(settings.BASE_DIR) / 'models' / 'efficient-yolov5l-obj365.pt',
    Path(settings.BASE_DIR) / 'models' / 'yolov5l.pt',
    Path(settings.BASE_DIR) / 'smartlabel' / 'apps' / 'algorithm' / 'efficientteacher-main' / 'models' / 'efficient-yolov5l-obj365.pt',
    Path(settings.BASE_DIR) / 'smartlabel' / 'apps' / 'algorithm' / 'efficientteacher-main' / 'models' / 'yolov5l.pt',
]
DETECTION_PRETRAINED_WEIGHT = Path(os.environ.get(
    'SMARTLABEL_EFFICIENTTEACHER_WEIGHTS',
    str(next((p for p in DETECTION_PRETRAINED_WEIGHT_CANDIDATES if p.exists()), DETECTION_PRETRAINED_WEIGHT_CANDIDATES[0])),
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
DETECTION_PLACEHOLDER_LABEL = 'obj'
DETECTION_COCO_CLASS_INDEX = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'boat': 8,
}
DEFAULT_TEXT_CLASSIFICATION_BERT_MODEL = Path(os.environ.get(
    'SMARTLABEL_TEXT_CLASSIFICATION_BERT_MODEL',
    str(Path(settings.BASE_DIR) / 'models' / 'chinese-roberta-wwm-ext'),
))


def _resolve_text_classification_bert_model():
    """优先用本地权重，本地缺失时回退到 HuggingFace 在线模型名。"""
    if DEFAULT_TEXT_CLASSIFICATION_BERT_MODEL.exists():
        return str(DEFAULT_TEXT_CLASSIFICATION_BERT_MODEL)
    return 'hfl/chinese-roberta-wwm-ext'


def _heartbeat_cache_key(task_id):
    """算法心跳在 Redis 中的 key。"""
    return f"task:{task_id}:algo_heartbeat"


def _remove_detection_placeholder_label(labels):
    """移除占位类别 'obj'：只要还存在真实类别，占位类别就不该进入训练。"""
    label_list = [str(label or '').strip() for label in labels if str(label or '').strip()]
    has_real_label = any(label != DETECTION_PLACEHOLDER_LABEL for label in label_list)
    if not has_real_label:
        return label_list
    return [label for label in label_list if label != DETECTION_PLACEHOLDER_LABEL]


def _set_algorithm_heartbeat(task_id, line):
    """记录算法子进程最新输出行与时间戳，供前端判断“是否卡死”。"""
    cache.set(
        _heartbeat_cache_key(task_id),
        {
            'at': datetime.now(timezone.utc).isoformat(),
            'line': (line or '')[:240],
        },
        timeout=24 * 3600,
    )


def clear_algorithm_heartbeat(task_id):
    """任务结束时清除心跳。"""
    cache.delete(_heartbeat_cache_key(task_id))


def _update_task_progress(task, progress, celery_task=None):
    """阶段性更新任务进度，避免进度长时间无变化。

    只允许进度单调递增，防止阶段回退导致前端进度条跳动。
    """
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
    """创建进度跟踪器函数。

    返回的闭包按“1% 变化或超过最小间隔”才写库，用于结果落库等内部循环中推进进度。
    """
    # 进度跟踪器状态
    progress_data = {
        'current': 0,
        'total': 0,
        'last_updated': time.time()
    }

    def update_progress(current, total):
        """按“1% 变化或超过最小间隔”写库，避免内部循环高频更新。"""
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


def run_subprocess_with_progress(cmd, task_id, total_files=None, progress_start=0, progress_end=100):
    """
    运行子进程并解析输出中的进度信息，更新任务进度。
    支持两种模式：
    1. 如果提供了 total_files，则会匹配 "正在处理" 等关键词来计算进度。
    2. 如果未提供 total_files，则会匹配 "处理进度: xx%" 来获取进度。
    进度可映射到指定区间，避免子步骤完成时把整个任务直接推进到 100%。
    """
    from celery import current_task
    import re
    
    try:
        task_obj = Task.objects.get(id=task_id)
        _set_algorithm_heartbeat(task_id, 'subprocess_started')
        
        # 子进程 stdout/stderr 合并为一路，统一按行解析；UTF-8 + replace 容错乱码
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            bufsize=1,
            encoding='utf-8', # 明确指定编码
            errors='replace',   # 替换无法解码的字符
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )
        
        # 子步骤进度映射到 [progress_start, progress_end] 区间，避免子步骤直接把整体推到 100%
        progress_start = int(max(0, min(progress_start, 99)))
        progress_end = int(max(progress_start, min(progress_end, 99)))

        def _stage_progress(raw_progress):
            """把子进程自身的 0-100% 线性映射到本阶段的区间。"""
            raw_progress = max(0.0, min(float(raw_progress), 100.0))
            return int(progress_start + (raw_progress / 100.0) * (progress_end - progress_start))

        # --- 模式B：基于百分比的正则 ---
        progress_pattern = re.compile(r'处理进度: (\d+)%')
        epoch_progress_pattern = re.compile(r'\b(?:Epoch|epoch)\s+(\d{1,4})/(\d{1,4})(?:\s|$)')
        
        # --- 模式A：基于文件计数的正则 ---
        file_progress_pattern = re.compile(r'Processing image:|正在处理图像:|\bDone\.') 

        current_progress = int(task_obj.progress or 0)
        processed_files = 0
        output_queue = queue.Queue()
        subprocess_started_at = time.time()
        last_synthetic_update_at = subprocess_started_at
        
        def _save_subprocess_progress(progress, reason, source_line=''):
            """写进度（只增不减），并同步到 Celery 状态。"""
            nonlocal current_progress, task_obj
            progress = int(max(0, min(progress, 100)))
            if progress <= current_progress:
                return False
            current_progress = progress
            task_obj.progress = progress
            task_obj.save(update_fields=['progress'])
            if current_task:
                current_task.update_state(state='PROGRESS', meta={'progress': progress})
            if source_line:
                logger.info(f"子进程输出 ({reason}): {source_line}")
            else:
                logger.info("子进程运行中，阶段进度推进到 %s%%", progress)
            return True

        # 单独线程读取 stdout，主线程用队列消费，避免管道阻塞
        def _read_subprocess_output():
            """后台线程：持续把子进程输出塞进队列，避免管道写满阻塞子进程。"""
            for output_line in iter(process.stdout.readline, ''):
                output_queue.put(output_line)

        output_thread = threading.Thread(target=_read_subprocess_output, daemon=True)
        output_thread.start()

        def _handle_subprocess_line(line):
            """解析单行输出：优先百分比，其次 epoch，最后文件计数。"""
            nonlocal processed_files
            line_processed_as_progress = False
            stripped_line = line.strip()
            safe_line = stripped_line.replace('\ufffd', '?')
            if safe_line:
                _set_algorithm_heartbeat(task_id, safe_line)

            # --- 优先匹配模式B (百分比) ---
            match_percent = progress_pattern.search(safe_line)
            if match_percent:
                try:
                    progress = _stage_progress(int(match_percent.group(1)))
                    
                    line_processed_as_progress = _save_subprocess_progress(
                        progress,
                        '进度更新',
                        safe_line,
                    )
                except (ValueError, TypeError) as e:
                    logger.error(f"解析进度百分比时出错: {str(e)}")
                return

            match_epoch = epoch_progress_pattern.search(safe_line)
            if match_epoch:
                try:
                    current_epoch = int(match_epoch.group(1))
                    max_epoch = int(match_epoch.group(2))
                    if max_epoch > 0 and current_epoch <= max_epoch:
                        progress = _stage_progress(((current_epoch + 1) / (max_epoch + 1)) * 100)
                        line_processed_as_progress = _save_subprocess_progress(
                            progress,
                            'epoch进度更新',
                            safe_line,
                        )
                except (ValueError, TypeError) as e:
                    logger.error(f"解析epoch进度时出错: {str(e)}")
                return

            # --- 如果模式B未匹配，且 total_files 有效，则尝试模式A (文件计数) ---
            if total_files and total_files > 0:
                match_file = file_progress_pattern.search(safe_line)
                if match_file:
                    processed_files += 1
                    raw_progress = min((processed_files / total_files) * 100, 99)
                    progress = _stage_progress(raw_progress)
                    
                    line_processed_as_progress = _save_subprocess_progress(
                        progress,
                        '文件计数更新',
                        safe_line,
                    )
                    return
                    
            # 如果以上两种进度模式都未匹配，则按通用信息记录一次
            if not line_processed_as_progress:
                logger.info(f"子进程输出: {safe_line}") 

        while process.poll() is None or not output_queue.empty():
            try:
                line = output_queue.get(timeout=0.5)
                _handle_subprocess_line(line)
                continue
            except queue.Empty:
                pass

            now = time.time()
            # 若子进程长时间不输出可解析的进度（如加载模型），按耗时缓慢推进，避免前端“假死”
            if progress_end > progress_start and now - last_synthetic_update_at >= 2:
                elapsed = now - subprocess_started_at
                synthetic_progress = min(
                    progress_end - 1,
                    progress_start + max(1, int(elapsed // 3)),
                )
                _save_subprocess_progress(synthetic_progress, '计时推进')
                last_synthetic_update_at = now

        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"子进程返回非零状态码: {return_code}")
            raise subprocess.CalledProcessError(return_code, cmd, output=None, stderr="subprocess exited with non-zero code")
        
        # 确保子步骤成功后推进到该阶段末端，整体任务完成由外层逻辑置 100。
        if task_obj.progress < progress_end:
            task_obj.progress = progress_end
            task_obj.save(update_fields=['progress'])
            if current_task:
                current_task.update_state(state='PROGRESS', meta={'progress': progress_end})

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
    """保存结果到数据库，并在落库阶段动态推进任务进度。

    先清空该任务旧结果，再按 CSV 分批 bulk_create；落库进度映射到 85-99%。
    """
    if not result_file_path.exists():
        logger.error(f"未找到结果文件: {result_file_path}")
        raise FileNotFoundError(f"未找到结果文件: {result_file_path}")

    # 任务类型 → 结果模型，决定用哪张表承接结果
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
        """把 CSV 中的置信度字符串安全转 float，空值或异常一律回退 default。"""
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


def _merge_manual_labels_if_needed(task):
    """如果存在人工标注文件，则与原始标注文件合并后返回新的 CSV 路径。

    合并规则：以文件名为键，人工标注覆盖原始标注；两者都无则报错提示先保存人工标注。
    """
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
    """文本任务人工补标清单的落盘路径。"""
    return Path(settings.MEDIA_ROOT) / 'results' / str(task.id) / 'manual_text_labels.csv'


@shared_task(
    bind=True,
    time_limit=7200,
    soft_time_limit=3600
)
def process_annotation_task(self, task_id):
    """处理半监督模式的任务 (修复 'glob' 错误并增加健壮性)

    这是所有标注任务的统一入口，按 task_type 分发到图像/文本/检测三条支线。
    """
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
        # 数据来自解压目录；有标注则走半监督训练，无任何标注则走“冷启动”只输出不确定性
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
                    """无任何标注时的冷启动打分：用灰度图信息熵近似不确定性（越乱越值得标注）。"""
                    try:
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
                logger.info(f"半监督任务 {task_id} 处理完成")
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
            run_subprocess_with_progress(
                cmd,
                task_id,
                progress_start=20,
                progress_end=80,
            )

            # 子进程结束，推进到80，后续由结果落库推进到100
            _update_task_progress(task, 80, self)

            if not result_file_path.exists():
                raise FileNotFoundError(f"图像分类结果文件不存在: {result_file_path}")
            _update_task_progress(task, 85, self)
            save_results_to_database(result_file_path, task, 'image-classification', update_progress)

        # ==================== 文本分类逻辑 ====================
        # 无标签：只按文本长度估计不确定性；有标签：合并人工标注后调用 CG3 文本分类算法
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
                run_subprocess_with_progress(
                    cmd,
                    task_id,
                    progress_start=30,
                    progress_end=85,
                )
                _update_task_progress(task, 80, self)

                task.refresh_from_db()

                if not result_file_path.exists():
                    raise FileNotFoundError(f"文本分类结果文件不存在: {result_file_path}")
                _update_task_progress(task, 85, self)
                save_results_to_database(result_file_path, task, 'text-classification', update_progress)

        # ==================== 目标检测逻辑 ====================
        # 最复杂的分支：解析多种来源标注 → 生成 YOLO 标签 → 训练/推理 → 写入闭环表
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
            declared_class_names = set()

            image_by_name = {p.name: p for p in image_files}

            def _canonical_detection_label(label):
                """把中文别名等归一化为统一英文类别名，保证类别一致。"""
                label_text = str(label or '').strip()
                if not label_text:
                    return DETECTION_PLACEHOLDER_LABEL
                return DETECTION_LABEL_ALIASES.get(label_text, label_text)

            def _default_detection_label():
                """只有一个真实类别时用它作默认类别，否则回退占位类别 'obj'。"""
                task_labels = _remove_detection_placeholder_label(
                    [_canonical_detection_label(label) for label in (task.label_list or [])]
                    + list(declared_class_names)
                )
                if len(task_labels) == 1:
                    return task_labels[0]
                return DETECTION_PLACEHOLDER_LABEL

            def _normalize_detection_xyxy(xmin, ymin, xmax, ymax, image_width, image_height):
                """把任意坐标系下的 xyxy 归一化为 0-1 的中心点+宽高；非法框返回 None。

                若四值本就落在 [0,1] 视为已归一化，否则按图像尺寸做归一化。
                """
                if xmax <= xmin or ymax <= ymin or image_width <= 0 or image_height <= 0:
                    return None

                values = [xmin, ymin, xmax, ymax]
                if all(0.0 <= value <= 1.0 for value in values):
                    x = (xmin + xmax) / 2.0
                    y = (ymin + ymax) / 2.0
                    w = xmax - xmin
                    h = ymax - ymin
                else:
                    x = (xmin + xmax) / 2.0 / image_width
                    y = (ymin + ymax) / 2.0 / image_height
                    w = (xmax - xmin) / image_width
                    h = (ymax - ymin) / image_height

                if w <= 0 or h <= 0:
                    return None
                return {
                    'x': max(0.0, min(1.0, x)),
                    'y': max(0.0, min(1.0, y)),
                    'w': max(0.0, min(1.0, w)),
                    'h': max(0.0, min(1.0, h)),
                }

            def _bounds_to_detection_boxes(raw_bounds, image_width, image_height):
                """把 "x1 y1 x2 y2 x1 y1 x2 y2 ..." 形式的框串解析成归一化框列表。"""
                if raw_bounds in (None, ''):
                    return []
                parts = str(raw_bounds).replace(',', ' ').split()
                try:
                    values = [float(part) for part in parts]
                except (TypeError, ValueError):
                    return []

                boxes = []
                for index in range(0, len(values) - 3, 4):
                    box = _normalize_detection_xyxy(
                        values[index],
                        values[index + 1],
                        values[index + 2],
                        values[index + 3],
                        image_width,
                        image_height,
                    )
                    if box:
                        boxes.append(box)
                return boxes

            def _detection_box_xyxy(box):
                """把归一化框（中心点+宽高）换算为裁剪到 [0,1] 的 xyxy 角点。"""
                x = float(box.get('x', 0))
                y = float(box.get('y', 0))
                w = float(box.get('w', 0))
                h = float(box.get('h', 0))
                return (
                    max(0.0, x - w / 2.0),
                    max(0.0, y - h / 2.0),
                    min(1.0, x + w / 2.0),
                    min(1.0, y + h / 2.0),
                )

            def _detection_box_iou(left, right):
                """计算两个归一化框的交并比（IoU），用于去重。"""
                lx1, ly1, lx2, ly2 = _detection_box_xyxy(left)
                rx1, ry1, rx2, ry2 = _detection_box_xyxy(right)
                inter_w = max(0.0, min(lx2, rx2) - max(lx1, rx1))
                inter_h = max(0.0, min(ly2, ry2) - max(ly1, ry1))
                inter_area = inter_w * inter_h
                left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
                right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
                union_area = left_area + right_area - inter_area
                return inter_area / union_area if union_area > 0 else 0.0

            def _dedupe_detection_boxes(boxes, iou_threshold=0.9):
                """同类别内去掉 IoU≥阈值 的重复框（保留先出现的）。"""
                deduped = []
                for box in boxes or []:
                    label = str(box.get('label') or '').strip()
                    if any(
                        str(kept.get('label') or '').strip() == label
                        and _detection_box_iou(box, kept) >= iou_threshold
                        for kept in deduped
                    ):
                        continue
                    deduped.append(box)
                return deduped

            def _append_coco_annotations(coco_data):
                """把 COCO 格式标注转成归一化框，追加到 labeled_map（按文件名索引）。"""
                categories = {
                    int(c.get('id')): _canonical_detection_label(c.get('name') or 'obj')
                    for c in coco_data.get('categories', [])
                    if c.get('id') is not None
                }
                declared_class_names.update(name for name in categories.values() if name)
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

            def _append_csv_annotations(csv_file_path):
                """把 CSV 标注（class/filename/bounds 等列）转成归一化框追加到 labeled_map。"""
                loaded_csv_boxes = 0
                with open(csv_file_path, 'r', encoding='utf-8', newline='') as cf:
                    reader = csv.DictReader(cf)
                    for row in reader:
                        raw_label = (row.get('class') or row.get('label') or row.get('name') or row.get('category') or '').strip()
                        if raw_label:
                            declared_class_names.add(_canonical_detection_label(raw_label))
                        fname = (row.get('filename') or row.get('image') or row.get('image_name') or '').strip()
                        if not fname:
                            continue
                        img_name = Path(fname).name
                        img_path = image_by_name.get(img_name)
                        if not img_path:
                            continue

                        with Image.open(img_path) as im:
                            iw, ih = im.size
                        if iw <= 0 or ih <= 0:
                            continue

                        if row.get('bounds'):
                            boxes = _bounds_to_detection_boxes(row.get('bounds'), iw, ih)
                        else:
                            try:
                                boxes = [
                                    _normalize_detection_xyxy(
                                        float(row.get('xmin', '')),
                                        float(row.get('ymin', '')),
                                        float(row.get('xmax', '')),
                                        float(row.get('ymax', '')),
                                        iw,
                                        ih,
                                    )
                                ]
                            except (TypeError, ValueError):
                                boxes = []
                            boxes = [box for box in boxes if box]

                        lbl = _canonical_detection_label(raw_label or _default_detection_label())
                        for box in boxes:
                            labeled_map.setdefault(img_name, []).append({
                                'label': lbl,
                                **box,
                            })
                            loaded_csv_boxes += 1
                return loaded_csv_boxes

            try:
                for names_file in list(Path(extract_dir).rglob('classes.txt')) + list(Path(extract_dir).rglob('obj.names')):
                    with open(names_file, 'r', encoding='utf-8') as nf:
                        for line in nf:
                            label_name = line.strip()
                            if label_name:
                                declared_class_names.add(_canonical_detection_label(label_name))
            except Exception as e:
                logger.warning("任务 %s 预扫描目标检测类别失败: %s", task_id, e)

            # 若上传了标注文件（可选），优先尝试解析 JSON(COCO) 或 CSV
            if task.label_file and Path(task.label_file.path).exists():
                label_file_path = Path(task.label_file.path)
                try:
                    if label_file_path.suffix.lower() == '.json':
                        with open(label_file_path, 'r', encoding='utf-8') as jf:
                            label_json = json.load(jf)
                        if isinstance(label_json, dict):
                            for shape in label_json.get('shapes', []) or []:
                                if isinstance(shape, dict) and shape.get('label'):
                                    declared_class_names.add(_canonical_detection_label(shape.get('label')))
                        if isinstance(label_json, dict) and 'images' in label_json and 'annotations' in label_json:
                            before_label_count = len(labeled_map)
                            _append_coco_annotations(label_json)
                            external_label_loaded = external_label_loaded or len(labeled_map) > before_label_count
                            logger.info("任务 %s 从上传标注 JSON %s 解析到 %s 张有标注图片", task_id, label_file_path.name, len(labeled_map))
                    elif label_file_path.suffix.lower() == '.csv':
                        loaded_csv_boxes = _append_csv_annotations(label_file_path)
                        external_label_loaded = external_label_loaded or loaded_csv_boxes > 0
                        logger.info("Loaded detection CSV labels for task %s from %s: %s images, %s boxes", task_id, label_file_path.name, len(labeled_map), loaded_csv_boxes)
                    elif label_file_path.suffix.lower() in {'.txt', '.names'}:
                        with open(label_file_path, 'r', encoding='utf-8') as lf:
                            for line in lf:
                                label_name = line.strip()
                                if label_name:
                                    declared_class_names.add(_canonical_detection_label(label_name))
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

            if not external_label_loaded:
                try:
                    csv_candidates = sorted(Path(extract_dir).rglob('*.csv'))
                    for csv_file in csv_candidates:
                        loaded_csv_boxes = _append_csv_annotations(csv_file)
                        external_label_loaded = external_label_loaded or loaded_csv_boxes > 0
                        if loaded_csv_boxes > 0:
                            logger.info(
                                "任务 %s 从压缩包 CSV %s 解析到 %s 张有标注图片，%s 个框",
                                task_id,
                                csv_file.name,
                                len(labeled_map),
                                loaded_csv_boxes,
                            )
                            break
                except Exception as e:
                    logger.warning("任务 %s 解析压缩包 CSV 标注失败，将继续使用人工标注流程: %s", task_id, e)

            try:
                for names_file in list(Path(extract_dir).rglob('classes.txt')) + list(Path(extract_dir).rglob('obj.names')):
                    with open(names_file, 'r', encoding='utf-8') as nf:
                        for line in nf:
                            label_name = line.strip()
                            if label_name:
                                declared_class_names.add(_canonical_detection_label(label_name))

                for json_file in list(Path(extract_dir).rglob('instances*.json')) + list(Path(extract_dir).rglob('annotations*.json')):
                    with open(json_file, 'r', encoding='utf-8') as jf:
                        label_json = json.load(jf)
                    if isinstance(label_json, dict):
                        for category in label_json.get('categories', []) or []:
                            if isinstance(category, dict) and category.get('name'):
                                declared_class_names.add(_canonical_detection_label(category.get('name')))
                        for shape in label_json.get('shapes', []) or []:
                            if isinstance(shape, dict) and shape.get('label'):
                                declared_class_names.add(_canonical_detection_label(shape.get('label')))
            except Exception as e:
                logger.warning("浠诲姟 %s 鎵弿鐩爣妫€娴嬪０鏄庣被鍒け璐? %s", task_id, e)

            db_manual_map = {}
            db_manual_confirmed_names = set()
            for ir in ImageResult.objects.filter(task=task).all():
                if ir.confidence == -1:
                    db_manual_confirmed_names.add(ir.image_name)
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
            if db_manual_map or db_manual_confirmed_names:
                logger.info(
                    "任务 %s 合并数据库人工确认样本 %s 张图片，其中 %s 张含目标框；训练标注总计 %s 张图片",
                    task_id,
                    len(db_manual_confirmed_names),
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
                    usable_labeled_map[image_name] = _dedupe_detection_boxes(usable_boxes)

            manual_negative_names = db_manual_confirmed_names - set(usable_labeled_map.keys())
            labeled_map = usable_labeled_map
            if manual_negative_names:
                logger.info(
                    "任务 %s 检测到 %s 张人工确认无目标图片，将作为目标检测负样本参与训练",
                    task_id,
                    len(manual_negative_names),
                )

            image_sequence_by_name = {p.name: index for index, p in enumerate(image_files, start=1)}

            def _accepted_manual_detection_annotations(boxes):
                """把外部标注框包装成“人工已确认”的标注结构（source=manual/decision=accepted）。"""
                annotations = []
                for box in boxes:
                    annotations.append({
                        'label': box.get('label') or '',
                        'x': box.get('x'),
                        'y': box.get('y'),
                        'w': box.get('w'),
                        'h': box.get('h'),
                        'shape_type': 'rectangle',
                        'source': 'manual',
                        'decision': 'accepted',
                        'sample_mode': 'uploaded_label',
                    })
                return annotations

            def _sync_external_detection_labels():
                """把外部标注同步为“人工已确认”样本（confidence=-1），并清掉同图模型预测。"""
                ResultModel = apps.get_model('data_management', 'ImageResult')
                synced_count = 0
                for image_name, boxes in labeled_map.items():
                    img_path = image_by_name.get(image_name)
                    if not img_path or not boxes:
                        continue
                    labels = [box.get('label') for box in boxes if box.get('label')]
                    primary_label = labels[0] if labels else ''
                    annotations = _accepted_manual_detection_annotations(boxes)
                    existing = ResultModel.objects.filter(task=task, image_name=image_name).order_by('-confidence', 'id').first()
                    if existing and existing.confidence == -1:
                        ResultModel.objects.filter(task=task, image_name=image_name).exclude(confidence=-1).delete()
                        continue
                    if existing:
                        existing.label = primary_label
                        existing.confidence = -1.0
                        existing.annotations = annotations
                        existing.status = 'verified'
                        existing.save(update_fields=['label', 'confidence', 'annotations', 'status'])
                    else:
                        ResultModel.objects.create(
                            task=task,
                            sequence_number=image_sequence_by_name.get(image_name, 1),
                            image_name=image_name,
                            image_path=str(img_path.resolve()),
                            label=primary_label,
                            confidence=-1.0,
                            annotations=annotations,
                            status='verified',
                        )
                    ResultModel.objects.filter(task=task, image_name=image_name).exclude(confidence=-1).delete()
                    synced_count += 1
                if synced_count:
                    logger.info("任务 %s 已将 %s 张外部检测标注同步为人工已确认样本", task_id, synced_count)

            # 如果 Task 中预设了 label_list，优先使用
            if task.label_list:
                class_list = []
                for label_name in task.label_list:
                    canonical_label = _canonical_detection_label(label_name)
                    if canonical_label not in class_list:
                        class_list.append(canonical_label)
                for label_name in sorted(declared_class_names | class_names):
                    canonical_label = _canonical_detection_label(label_name)
                    if canonical_label not in class_list:
                        class_list.append(canonical_label)
            else:
                class_list = sorted(list(declared_class_names | class_names))

            class_list = _remove_detection_placeholder_label(class_list)
            if not class_list:
                class_list = [DETECTION_PLACEHOLDER_LABEL]

            if list(task.label_list or []) != class_list:
                task.label_list = class_list
                task.save(update_fields=['label_list'])

            class2id = {n: i for i, n in enumerate(class_list)}
            _sync_external_detection_labels()
            detection_review_context = build_detection_review_context(
                image_by_name,
                labeled_map,
                class_list,
            )
            detection_review_context = merge_positive_index_context(task, detection_review_context)
            review_queue_version = get_next_queue_version(task)
            total_images = max(len(image_files), 1)

            def _update_detection_stage_progress(current, total, start, end):
                """把 [0,total] 的当前进度线性映射到 [start,end] 并写库。"""
                if total <= 0:
                    _update_task_progress(task, end, self)
                    return
                progress = start + round((current / total) * (end - start))
                _update_task_progress(task, progress, self)

            def _should_update_detection_progress(index, total):
                """按约 1/40 的粒度写进度，避免逐条循环频繁写库。"""
                if total <= 0:
                    return False
                batch_size = max(1, math.ceil(total / 40))
                return index == total or index % batch_size == 0

            def _label_path_for_image(img_path):
                """由图片路径推导对应的 YOLO label 文件路径（images/ → labels/）。"""
                parts = list(img_path.parts)
                if 'images' in parts:
                    idx = parts.index('images')
                    parts[idx] = 'labels'
                    return Path(*parts).with_suffix('.txt')
                return img_path.parent / 'labels' / f"{img_path.stem}.txt"

            # 为每张图片生成 YOLO 格式的 label 文件（放在解压目录旁，跟图片同目录）
            for index, img_path in enumerate(image_files, start=1):
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
                        cid = class2id.get(lbl)
                        if cid is None:
                            continue
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
                if _should_update_detection_progress(index, total_images):
                    _update_detection_stage_progress(index, total_images, 10, 20)

            _update_task_progress(task, 20, self)

            # 生成 train/val/target 列表
            labeled_images = []
            positive_labeled_images = []
            for p in image_files:
                label_path = _label_path_for_image(p)
                is_positive = label_path.exists() and label_path.stat().st_size > 0
                is_manual_negative = p.name in manual_negative_names
                if is_positive:
                    positive_labeled_images.append(p)
                if is_positive or is_manual_negative:
                    labeled_images.append(p)
            unlabeled_images = [p for p in image_files if p not in labeled_images]

            detect_py = Path(settings.BASE_DIR) / 'smartlabel' / 'apps' / 'algorithm' / 'efficientteacher-main' / 'detect.py'
            detect_source = extract_dir / 'images' if (extract_dir / 'images').exists() else extract_dir

            def _detection_review_result(annotations, image_path, pre_filter_count=0, post_filter_count=0):
                """调用主动学习评分，返回 (复核价值分, 低置信框数, 明细)。"""
                return detection_review_value_detail(
                    annotations,
                    image_path,
                    review_context=detection_review_context,
                    pre_filter_count=pre_filter_count,
                    post_filter_count=post_filter_count,
                )

            def _detection_review_value(annotations, image_path, pre_filter_count=0, post_filter_count=0):
                """只取 (复核价值分, 低置信框数) 两个值的简化版评分。"""
                score, low_confidence_count, _detail = _detection_review_result(
                    annotations,
                    image_path,
                    pre_filter_count=pre_filter_count,
                    post_filter_count=post_filter_count,
                )
                return score, low_confidence_count

            def _build_empty_detection_rows(progress_start=80, progress_end=88):
                """为每张图生成“无预测框”的占位结果行（仍带复核价值分）。"""
                rows = []
                for seq, img_path in enumerate(image_files, start=1):
                    review_score, low_confidence_count, review_detail = _detection_review_result([], img_path)
                    rows.append({
                        'sequence_number': seq,
                        'image_name': img_path.name,
                        'image_path': str(img_path.resolve()),
                        'annotations': [],
                        'confidence': 0.0,
                        'uncertainty_score': review_score,
                        'low_confidence_count': low_confidence_count,
                        'detection_count': 0,
                        'status': 'unverified',
                        'review_detail': review_detail,
                    })
                    if _should_update_detection_progress(seq, total_images):
                        _update_detection_stage_progress(seq, total_images, progress_start, progress_end)
                return rows

            def _coco_id_to_label_map():
                """把本任务的类别映射为 COCO 官方类号 → 类别名，供基础模型推理时 --classes 过滤。"""
                coco_id_to_label = {}
                for label in class_list:
                    normalized_label = DETECTION_LABEL_ALIASES.get(str(label).strip(), str(label).strip())
                    coco_id = DETECTION_COCO_CLASS_INDEX.get(normalized_label)
                    if coco_id is not None:
                        coco_id_to_label[coco_id] = normalized_label
                return coco_id_to_label

            def _collect_detection_rows_from_labels(pred_labels_dir, id_to_label, sample_mode, progress_start=80, progress_end=88):
                """读取模型输出的 YOLO label 文件，按置信度阈值过滤并生成结果行。

                返回 (rows, 原始框总数, 被过滤框数)，同时为每张图计算复核价值分。
                """
                rows = []
                raw_count = 0
                filtered_count = 0
                for seq, img_path in enumerate(image_files, start=1):
                    label_file = pred_labels_dir / (img_path.stem + '.txt')
                    annotations = []
                    max_conf = 0.0
                    image_raw_count = 0
                    if label_file.exists():
                        with open(label_file, 'r', encoding='utf-8') as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if not parts or len(parts) < 5:
                                    continue
                                cid = int(float(parts[0]))
                                label = id_to_label.get(cid)
                                if not label:
                                    continue
                                x = float(parts[1])
                                y = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                conf = float(parts[5]) if len(parts) >= 6 else 0.0
                                raw_count += 1
                                image_raw_count += 1
                                if conf < DETECTION_STORE_CONF_THRES:
                                    filtered_count += 1
                                    continue
                                annotations.append({
                                    'label': label,
                                    'x': x,
                                    'y': y,
                                    'w': w,
                                    'h': h,
                                    'confidence': conf,
                                    'shape_type': 'rectangle',
                                    'source': 'model',
                                    'decision': 'pending',
                                    'sample_mode': sample_mode,
                                })
                        annotations.sort(key=lambda item: float(item.get('confidence') or 0.0), reverse=True)
                        annotations = _dedupe_detection_boxes(annotations)
                        annotations = annotations[:DETECTION_MAX_BOXES_PER_IMAGE]
                        max_conf = max((float(item.get('confidence') or 0.0) for item in annotations), default=0.0)

                    uncertainty_score, low_confidence_count, review_detail = _detection_review_result(
                        annotations,
                        img_path,
                        pre_filter_count=image_raw_count,
                        post_filter_count=len(annotations),
                    )
                    rows.append({
                        'sequence_number': seq,
                        'image_name': img_path.name,
                        'image_path': str(img_path.resolve()),
                        'annotations': annotations,
                        'confidence': max_conf,
                        'uncertainty_score': uncertainty_score,
                        'low_confidence_count': low_confidence_count,
                        'detection_count': len(annotations),
                        'status': 'unverified',
                        'review_detail': review_detail,
                    })
                    if _should_update_detection_progress(seq, total_images):
                        _update_detection_stage_progress(seq, total_images, progress_start, progress_end)
                return rows, raw_count, filtered_count

            def _run_pretrained_detection_rows(reason, subprocess_start=20, subprocess_end=80, collect_start=80, collect_end=88):
                """用基础预训练权重（COCO/Obj365）做冷启动推理，产出初始候选框。

                仅在正样本不足（冷启动/few-shot）或微调结果过弱时使用。
                """
                if not DETECTION_PRETRAINED_WEIGHT.exists():
                    logger.warning("目标检测基础预训练权重不存在，无法执行 %s: %s", reason, DETECTION_PRETRAINED_WEIGHT)
                    return _build_empty_detection_rows(collect_start, collect_end), 0, 0
                coco_id_to_label = _coco_id_to_label_map()
                if not coco_id_to_label:
                    logger.info("当前目标检测标签不在 COCO 基础类别中，跳过基础模型冷启动推理: labels=%s", class_list)
                    return _build_empty_detection_rows(collect_start, collect_end), 0, 0

                pred_dir = Path(result_dir) / 'pred'
                if pred_dir.exists():
                    shutil.rmtree(pred_dir)
                cmd = [
                    'conda', 'run', '--no-capture-output', '-n', 'efficientteacher', 'python',
                    str(detect_py),
                    '--weights', str(DETECTION_PRETRAINED_WEIGHT),
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
                    '--classes',
                    *[str(i) for i in sorted(coco_id_to_label.keys())],
                ]
                logger.info(
                    "目标检测基础预训练推理: reason=%s, weight=%s, label_map=%s",
                    reason,
                    DETECTION_PRETRAINED_WEIGHT,
                    coco_id_to_label,
                )
                run_subprocess_with_progress(
                    cmd,
                    task_id,
                    total_files=total_images,
                    progress_start=subprocess_start,
                    progress_end=subprocess_end,
                )
                pred_labels_dir = pred_dir / 'labels'
                if not pred_labels_dir.exists():
                    return _build_empty_detection_rows(collect_start, collect_end), 0, 0
                return _collect_detection_rows_from_labels(
                    pred_labels_dir,
                    coco_id_to_label,
                    'base_pretrained_prediction',
                    collect_start,
                    collect_end,
                )

            def _persist_detection_rows(rows, mode):
                """把检测结果行写入 result.csv，并落库到 ImageResult（保留人工已确认样本）。"""
                result_dir.mkdir(parents=True, exist_ok=True)
                with open(result_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(
                        f,
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
                    for index, row in enumerate(rows, start=1):
                        csv_row = row.copy()
                        csv_row.pop('review_detail', None)
                        csv_row['annotations'] = json.dumps(csv_row['annotations'], ensure_ascii=False)
                        writer.writerow(csv_row)
                        if _should_update_detection_progress(index, len(rows)):
                            _update_detection_stage_progress(index, len(rows), 88, 91)

                ResultModel = apps.get_model('data_management', 'ImageResult')
                manual_image_names = set(
                    ResultModel.objects
                    .filter(task=task, confidence=-1)
                    .values_list('image_name', flat=True)
                )
                with transaction.atomic():
                    ResultModel.objects.filter(task=task).exclude(confidence=-1).delete()
                    instances = []
                    for index, row in enumerate(rows, start=1):
                        if row['image_name'] in manual_image_names:
                            continue
                        instances.append(ResultModel(
                            task=task,
                            sequence_number=row['sequence_number'],
                            image_name=row['image_name'],
                            image_path=row['image_path'],
                            label='',
                            confidence=row['confidence'],
                            annotations=row['annotations'],
                            status=row['status'],
                        ))
                        if _should_update_detection_progress(index, len(rows)):
                            _update_detection_stage_progress(index, len(rows), 91, 94)
                    if instances:
                        ResultModel.objects.bulk_create(instances)
                    if manual_image_names:
                        ResultModel.objects.filter(task=task, image_name__in=manual_image_names).exclude(confidence=-1).delete()
                persist_review_queue_items(
                    task,
                    rows,
                    review_queue_version,
                    progress_callback=lambda current, total: _update_detection_stage_progress(current, total, 94, 95),
                )

                _update_task_progress(task, 95, self)
                task.progress = 100
                task.status = 'completed'
                task.save(update_fields=['progress', 'status'])
                logger.info(
                    "目标检测任务 %s 已完成 %s，写入 %s 条候选样本，其中 %s 条包含预测框",
                    task_id,
                    mode,
                    len(rows),
                    sum(1 for row in rows if row['annotations']),
                )

            # 分支1：正样本不足 → 不训练，直接用基础预训练模型冷启动出候选框
            if len(positive_labeled_images) < DETECTION_MIN_TRAIN_IMAGES:
                active_mode = 'cold_start' if not positive_labeled_images else 'few_shot_active_learning'
                rows, raw_count, filtered_count = _run_pretrained_detection_rows(
                    active_mode
                )
                logger.info(
                    "目标检测正样本不足，跳过微调训练: positive_labeled=%s, labeled_total=%s, min_required=%s, raw_boxes=%s, filtered_low_confidence=%s",
                    len(positive_labeled_images),
                    len(labeled_images),
                    DETECTION_MIN_TRAIN_IMAGES,
                    raw_count,
                    filtered_count,
                )
                _persist_detection_rows(rows, active_mode)
                return {
                    "status": "success",
                    "task_id": task_id,
                    "mode": active_mode,
                }

            # 分支2：完全没有可用标注 → 生成全空框冷启动样本，等人工标注后重跑
            if not labeled_images:
                result_rows = []
                for seq, img_path in enumerate(image_files, start=1):
                    review_score, low_confidence_count, review_detail = _detection_review_result([], img_path)
                    result_rows.append({
                        'sequence_number': seq,
                        'image_name': img_path.name,
                        'image_path': str(img_path.resolve()),
                        'annotations': [],
                        'confidence': 0.0,
                        'uncertainty_score': review_score,
                        'low_confidence_count': low_confidence_count,
                        'detection_count': 0,
                        'status': 'unverified',
                        'review_detail': review_detail,
                    })

                result_dir.mkdir(parents=True, exist_ok=True)
                with open(result_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(
                        f,
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
                    for row in result_rows:
                        csv_row = row.copy()
                        csv_row.pop('review_detail', None)
                        csv_row['annotations'] = json.dumps(csv_row['annotations'], ensure_ascii=False)
                        writer.writerow(csv_row)

                ResultModel = apps.get_model('data_management', 'ImageResult')
                with transaction.atomic():
                    ResultModel.objects.filter(task=task).delete()
                    ResultModel.objects.bulk_create([
                        ResultModel(
                            task=task,
                            sequence_number=row['sequence_number'],
                            image_name=row['image_name'],
                            image_path=row['image_path'],
                            label='',
                            confidence=row['confidence'],
                            annotations=row['annotations'],
                            status=row['status'],
                        )
                        for row in result_rows
                    ])
                persist_review_queue_items(
                    task,
                    result_rows,
                    review_queue_version,
                    progress_callback=lambda current, total: _update_detection_stage_progress(current, total, 94, 95),
                )

                _update_task_progress(task, 95, self)
                task.progress = 100
                task.status = 'completed'
                task.save(update_fields=['progress', 'status'])
                logger.info(
                    "目标检测任务 %s 未检测到可用标注，已生成 %s 条冷启动待标注样本。",
                    task_id,
                    len(result_rows),
                )
                return {"status": "success", "task_id": task_id, "mode": "cold_start"}

            # 简单划分 val（10% 的有标签样本）
            # 只有 1 张时 train/val 都用它，避免 val 为空导致上游报错
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
                """写出数据集清单 txt；with_label 时每行追加对应 label 文件路径（YOLO 训练用）。"""
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
            if not DETECTION_PRETRAINED_WEIGHT.exists():
                raise FileNotFoundError(
                    f"目标检测固定预训练权重不存在: {DETECTION_PRETRAINED_WEIGHT}。"
                    "请先放置平台只读基础权重，可使用 models/efficient-yolov5l-obj365.pt "
                    "或 models/yolov5l.pt；目标检测任务只允许在该权重上进行任务级微调。"
                )

            cfg_lines = []
            cfg_lines.append(f"project: '{result_dir}'")
            cfg_lines.append(f"name: '{cfg_name}'")
            cfg_lines.append(f"epochs: {DETECTION_TRAIN_EPOCHS}")
            cfg_lines.append("adam: False")
            cfg_lines.append(f"freeze_layer_num: {DETECTION_FREEZE_LAYER_NUM}")
            pretrained_weight = str(DETECTION_PRETRAINED_WEIGHT)
            cfg_lines.append(f"weights: '{pretrained_weight}'")
            logger.info(
                "目标检测训练使用固定预训练权重: %s；冻结前 %s 个网络层，仅生成任务输出权重，不会覆盖基础权重",
                pretrained_weight,
                DETECTION_FREEZE_LAYER_NUM,
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
            cfg_lines.append(f"  batch_size: {DETECTION_TRAIN_BATCH_SIZE}")
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
            run_subprocess_with_progress(
                cmd,
                task_id,
                progress_start=40,
                progress_end=75,
            )

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
            if class_list:
                detect_cmd.extend(['--classes', *[str(i) for i in range(len(class_list))]])
            logger.info(
                "object detection predict args: source=%s, classes=%s, conf_thres=%s, store_conf_thres=%s, iou_thres=%s, max_det=%s",
                detect_source,
                class_list,
                DETECTION_PREDICT_CONF_THRES,
                DETECTION_STORE_CONF_THRES,
                DETECTION_PREDICT_IOU_THRES,
                DETECTION_PREDICT_MAX_DET,
            )
            pred_dir = Path(result_dir) / 'pred'
            if pred_dir.exists():
                shutil.rmtree(pred_dir)
            run_subprocess_with_progress(
                detect_cmd,
                task_id,
                total_files=total_images,
                progress_start=75,
                progress_end=82,
            )

            # 解析 detect 输出 labels，生成 result.csv 并落库
            pred_labels_dir = pred_dir / 'labels'
            if not pred_labels_dir.exists():
                raise FileNotFoundError(f"目标检测推理未生成标签目录: {pred_labels_dir}")

            results_rows = []
            seq = 0
            raw_predicted_box_count = 0
            filtered_low_confidence_count = 0

            for img_path in image_files:
                seq += 1
                img_name = img_path.name
                rel_path = str(img_path.resolve())
                label_file = pred_labels_dir / (img_path.stem + '.txt')
                annotations = []
                max_conf = 0.0
                image_raw_count = 0
                if label_file.exists():
                    with open(label_file, 'r', encoding='utf-8') as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            if len(parts) >= 5:
                                cid = int(float(parts[0]))
                                if cid < 0 or cid >= len(class_list):
                                    logger.debug(
                                        "跳过不属于当前任务标签集的目标检测类别: task=%s image=%s cid=%s labels=%s",
                                        task_id,
                                        img_name,
                                        cid,
                                        class_list,
                                    )
                                    continue
                                x = float(parts[1])
                                y = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                conf = float(parts[5]) if len(parts) >= 6 else 0.0
                                raw_predicted_box_count += 1
                                image_raw_count += 1
                                if conf < DETECTION_STORE_CONF_THRES:
                                    filtered_low_confidence_count += 1
                                    continue
                                annotations.append({
                                    'label': class_list[cid],
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
                    annotations = _dedupe_detection_boxes(annotations)
                    annotations = annotations[:DETECTION_MAX_BOXES_PER_IMAGE]
                    max_conf = max((float(item.get('confidence') or 0.0) for item in annotations), default=0.0)

                uncertainty_score, low_confidence_count, review_detail = _detection_review_result(
                    annotations,
                    img_path,
                    pre_filter_count=image_raw_count,
                    post_filter_count=len(annotations),
                )
                results_rows.append({
                    'sequence_number': seq,
                    'image_name': img_name,
                    'image_path': rel_path,
                    'annotations': annotations,
                    'confidence': max_conf,
                    'uncertainty_score': uncertainty_score,
                    'low_confidence_count': low_confidence_count,
                    'detection_count': len(annotations),
                    'status': 'unverified',
                    'review_detail': review_detail,
                })
                if _should_update_detection_progress(seq, total_images):
                    _update_detection_stage_progress(seq, total_images, 82, 88)

            # 将结果写入数据库（ImageResult）
            predicted_image_count = sum(1 for row in results_rows if row['annotations'])
            predicted_box_count = sum(len(row['annotations']) for row in results_rows)
            logger.info(
                "object detection predict finished: %s/%s images with boxes, %s boxes stored, %s raw boxes, %s low-confidence boxes filtered",
                predicted_image_count,
                len(results_rows),
                predicted_box_count,
                raw_predicted_box_count,
                filtered_low_confidence_count,
            )
            min_fine_tuned_images = max(1, int(len(results_rows) * DETECTION_MIN_FINE_TUNED_IMAGE_RATIO))
            fine_tuned_result_is_weak = (
                predicted_box_count < DETECTION_MIN_FINE_TUNED_BOXES
                or predicted_image_count < min_fine_tuned_images
            )
            # 分支3：微调结果过弱（框太少/覆盖图片太少）→ 用基础预训练权重兜底重跑一次推理
            if fine_tuned_result_is_weak and DETECTION_PRETRAINED_WEIGHT.exists() and DETECTION_PRETRAINED_WEIGHT.name.lower().startswith('yolov5'):
                coco_id_to_label = {}
                for label in class_list:
                    normalized_label = DETECTION_LABEL_ALIASES.get(str(label).strip(), str(label).strip())
                    coco_id = DETECTION_COCO_CLASS_INDEX.get(normalized_label)
                    if coco_id is not None:
                        coco_id_to_label[coco_id] = normalized_label
                if coco_id_to_label:
                    logger.warning(
                        "目标检测微调结果低于可靠阈值，将使用基础预训练模型兜底: images=%s/%s, boxes=%s, min_images=%s, min_boxes=%s",
                        predicted_image_count,
                        len(results_rows),
                        predicted_box_count,
                        min_fine_tuned_images,
                        DETECTION_MIN_FINE_TUNED_BOXES,
                    )
                    logger.warning(
                        "目标检测任务微调权重没有生成候选框，改用基础 YOLOv5 COCO 预训练权重兜底推理: weight=%s, label_map=%s",
                        DETECTION_PRETRAINED_WEIGHT,
                        coco_id_to_label,
                    )
                    if pred_dir.exists():
                        shutil.rmtree(pred_dir)
                    fallback_cmd = [
                        'conda', 'run', '--no-capture-output', '-n', 'efficientteacher', 'python',
                        str(detect_py),
                        '--weights', str(DETECTION_PRETRAINED_WEIGHT),
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
                        '--classes',
                        *[str(i) for i in sorted(coco_id_to_label.keys())],
                    ]
                    run_subprocess_with_progress(
                        fallback_cmd,
                        task_id,
                        total_files=total_images,
                        progress_start=88,
                        progress_end=92,
                    )
                    pred_labels_dir = pred_dir / 'labels'
                    if not pred_labels_dir.exists():
                        raise FileNotFoundError(f"目标检测基础预训练兜底推理未生成标签目录: {pred_labels_dir}")

                    results_rows = []
                    raw_predicted_box_count = 0
                    filtered_low_confidence_count = 0
                    for seq, img_path in enumerate(image_files, start=1):
                        img_name = img_path.name
                        rel_path = str(img_path.resolve())
                        label_file = pred_labels_dir / (img_path.stem + '.txt')
                        annotations = []
                        max_conf = 0.0
                        image_raw_count = 0
                        if label_file.exists():
                            with open(label_file, 'r', encoding='utf-8') as lf:
                                for line in lf:
                                    parts = line.strip().split()
                                    if not parts or len(parts) < 5:
                                        continue
                                    cid = int(float(parts[0]))
                                    label = coco_id_to_label.get(cid)
                                    if not label:
                                        continue
                                    x = float(parts[1])
                                    y = float(parts[2])
                                    w = float(parts[3])
                                    h = float(parts[4])
                                    conf = float(parts[5]) if len(parts) >= 6 else 0.0
                                    raw_predicted_box_count += 1
                                    image_raw_count += 1
                                    if conf < DETECTION_STORE_CONF_THRES:
                                        filtered_low_confidence_count += 1
                                        continue
                                    annotations.append({
                                        'label': label,
                                        'x': x,
                                        'y': y,
                                        'w': w,
                                        'h': h,
                                        'confidence': conf,
                                        'shape_type': 'rectangle',
                                        'source': 'model',
                                        'decision': 'pending',
                                        'sample_mode': 'base_pretrained_prediction',
                                    })
                            annotations.sort(key=lambda item: float(item.get('confidence') or 0.0), reverse=True)
                            annotations = _dedupe_detection_boxes(annotations)
                            annotations = annotations[:DETECTION_MAX_BOXES_PER_IMAGE]
                            max_conf = max((float(item.get('confidence') or 0.0) for item in annotations), default=0.0)

                        uncertainty_score, low_confidence_count, review_detail = _detection_review_result(
                            annotations,
                            img_path,
                            pre_filter_count=image_raw_count,
                            post_filter_count=len(annotations),
                        )
                        results_rows.append({
                            'sequence_number': seq,
                            'image_name': img_name,
                            'image_path': rel_path,
                            'annotations': annotations,
                            'confidence': max_conf,
                            'uncertainty_score': uncertainty_score,
                            'low_confidence_count': low_confidence_count,
                            'detection_count': len(annotations),
                            'status': 'unverified',
                            'review_detail': review_detail,
                        })
                        if _should_update_detection_progress(seq, total_images):
                            _update_detection_stage_progress(seq, total_images, 92, 94)
                    predicted_image_count = sum(1 for row in results_rows if row['annotations'])
                    predicted_box_count = sum(len(row['annotations']) for row in results_rows)
                    logger.info(
                        "object detection pretrained fallback finished: %s/%s images with boxes, %s boxes stored, %s raw boxes, %s low-confidence boxes filtered",
                        predicted_image_count,
                        len(results_rows),
                        predicted_box_count,
                        raw_predicted_box_count,
                        filtered_low_confidence_count,
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
                for index, row in enumerate(results_rows, start=1):
                    csv_row = row.copy()
                    csv_row.pop('review_detail', None)
                    csv_row['annotations'] = json.dumps(csv_row['annotations'], ensure_ascii=False)
                    writer.writerow(csv_row)
                    if _should_update_detection_progress(index, len(results_rows)):
                        _update_detection_stage_progress(index, len(results_rows), 88, 91)

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
                for index, r in enumerate(results_rows, start=1):
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
                    if _should_update_detection_progress(index, len(results_rows)):
                        _update_detection_stage_progress(index, len(results_rows), 91, 94)
                if instances:
                    ResultModel.objects.bulk_create(instances)
                if manual_image_names:
                    ResultModel.objects.filter(task=task, image_name__in=manual_image_names).exclude(confidence=-1).delete()
            persist_review_queue_items(
                task,
                results_rows,
                review_queue_version,
                progress_callback=lambda current, total: _update_detection_stage_progress(current, total, 94, 95),
            )

            _update_task_progress(task, 90, self)
            # 最终推进到完成状态由外层逻辑处理

        # --- 任务完成 ---
        task.progress = 100
        task.status = 'completed'
        task.save()
        
        logger.info(f"半监督任务 {task_id} 处理完成")
        return {"status": "success", "task_id": task_id}
        
    except (FileNotFoundError, ValueError) as e:
         # 可预期的数据/环境问题：标记失败并记录原因，不做重试
         logger.error(f"处理半监督任务 {task_id} 时发生数据校验错误: {str(e)}")
         task.status = 'failed'
         task.error_log = f"数据校验失败: {e}"
         task.save(update_fields=['status', 'error_log'])
         return {"status": "failed", "reason": str(e)}

    except Exception as e:
        # 未知异常同样标记失败；用 Ignore 阻止 Celery 自动重试，避免反复占用 Worker
        logger.error(f"处理半监督任务 {task_id} 时发生未知错误: {e}", exc_info=True)
        task.status = 'failed'
        task.error_log = f"内部服务器错误: {e}"
        task.save(update_fields=['status', 'error_log'])
        # 不再自动重试，避免训练脚本持续失败时反复占用 worker。
        raise Ignore()
