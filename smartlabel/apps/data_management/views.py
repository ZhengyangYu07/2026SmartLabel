import codecs
import csv
import hashlib
import logging
import os
import shutil
import signal
import uuid
import zipfile
# 引入处理 .rar 和 .7z 的库
import rarfile # 可能需要安装: pip install rarfile (且系统需安装 unrar)
import py7zr   # 可能需要安装: pip install py7zr
import io      # 用于内存文件操作
from pathlib import Path
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError, PermissionDenied
from django.core.paginator import Paginator, EmptyPage
from django.core.cache import cache
from django.db.models import F, Case, When, Value, Q, Count, FloatField, Exists, OuterRef, BooleanField
from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from io import StringIO
from kombu.utils import json
from pathlib import Path
from django.db.models.functions import Lower
from django.db import transaction
from functools import reduce
import operator
import urllib.parse
from redis import Redis
from django.urls import reverse

# 从现有Celery配置导入
from celery import current_app as celery_app
from celery.exceptions import OperationalError

from .models import Task, ImageResult, TextResult, FavoriteTask
from .tasks import process_annotation_task, process_pretrained_task
from .utils import process
from .utils.config import TASK_CONFIG, TaskService

logger = logging.getLogger(__name__)


def _get_configured_queue_names():
    """获取当前 Celery 配置中声明的队列名称。"""
    queue_names = set()
    queues = getattr(celery_app.conf, 'task_queues', None)

    if queues:
        for queue in queues:
            name = getattr(queue, 'name', None)
            if name:
                queue_names.add(name)

    default_queue = getattr(celery_app.conf, 'task_default_queue', None)
    if default_queue:
        queue_names.add(default_queue)

    if not queue_names:
        queue_names = {'default'}

    return queue_names


def _remove_task_message_from_broker(celery_task_id):
    """尽量从 Redis broker 队列中移除目标任务消息（仅针对尚未被 worker 消费的消息）。"""
    if not celery_task_id:
        return 0

    broker_url = celery_app.conf.broker_url
    if not broker_url or not str(broker_url).startswith('redis://'):
        logger.info("Broker 不是 Redis，跳过按 task_id 定向清理队列消息")
        return 0

    removed_count = 0
    redis_client = Redis.from_url(broker_url)

    for queue_name in _get_configured_queue_names():
        try:
            queue_messages = redis_client.lrange(queue_name, 0, -1)
            for message in queue_messages:
                payload = message.decode('utf-8', errors='ignore')
                if celery_task_id in payload:
                    removed_count += redis_client.lrem(queue_name, 1, message)
        except Exception as e:
            logger.warning("清理队列 %s 中的任务消息失败: %s", queue_name, e)

    return removed_count

# 图片分类允许的扩展名
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

# 文本分类允许的扩展名
ALLOWED_TEXT_EXTENSIONS = ['.csv']

# --- 辅助函数：将前端中文状态映射为后端英文状态 ---
STATUS_MAP_FRONTEND_TO_DB = {
    '已校验': 'verified',
    '未校验': 'unverified',
    # '已标注' 在数据库中不是一个 status 字段的值，而是通过 confidence=-1 识别的
    '已标注': 'manual_annotation_flag', # 仅用于表示该类型，实际查询是 confidence=-1
}

STATUS_MAP_DB_TO_FRONTEND = {
    'verified': '已校验',
    'unverified': '未校验',
}


def _get_label_verification_stats(task, result_model, relation_field):
    """统计任务整体的校验状态数据，并保留人工标注数量。"""
    task_queryset = result_model.objects.filter(**{relation_field: task})
    manual_queryset = task_queryset.filter(confidence=-1)

    verified_count = task_queryset.filter(status='verified').count()
    total_count = task_queryset.count()
    manual_annotated_count = manual_queryset.count()
    
    # 未校验 = confidence=-1 的数据数量 - 已校验
    unverified_count = manual_annotated_count - verified_count

    verification_progress = 0
    if total_count > 0:
        verification_progress = round((verified_count / total_count) * 100, 1)

    return {
        'manual_annotated_count': manual_annotated_count,
        'total_labeled_count': manual_annotated_count,
        'verified_count': verified_count,
        'unverified_count': unverified_count,
        'total_count': total_count,
        'total_verifiable_count': total_count,
        'verification_progress': verification_progress,
    }


def _check_compressed_file_contents(file_list, task_type):
    """
    检查压缩包内文件的类型是否符合任务类型。
    file_list: 压缩包内所有文件和目录的名称列表。
    task_type: 任务类型 ('image-classification' 或 'text-classification')。
    """
    if not file_list:
        return {'status': 'error', 'message': '压缩包内没有文件。'}

    # 过滤掉目录和特殊文件（如macOS的__MACOSX）
    actual_files = [
        f for f in file_list
        if not f.endswith('/') and not f.endswith('\\') and not f.startswith(('__MACOSX/', '.DS_Store', '.git/', '.svn/', '._')) and f.strip() # 过滤空文件名
    ]

    if not actual_files:
         return {'status': 'error', 'message': '压缩包内没有有效数据文件。请确保压缩包中包含可识别的文件。'}

    if task_type == 'image-classification':
        # 检查所有文件是否都是图片格式
        for file_name in actual_files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                return {'status': 'error', 'message': f'图像分类任务中检测到非图片文件: "{file_name}"。请确保压缩包只包含支持的图片文件 ({", ".join(ALLOWED_IMAGE_EXTENSIONS)})。'}
        return {'status': 'success', 'message': '数据文件内容符合图像分类任务要求。'}

    elif task_type == 'text-classification':
        # 过滤出所有CSV文件
        csv_files = [f for f in actual_files if os.path.splitext(f)[1].lower() == '.csv']
        
        # 过滤出所有非CSV文件 (排除系统文件)
        other_non_system_files = [
            f for f in actual_files
            if os.path.splitext(f)[1].lower() != '.csv' and
            not f.startswith(('__MACOSX/', '.DS_Store', '.git/', '.svn/', '._')) 
        ]

        if len(csv_files) == 0:
            return {'status': 'error', 'message': '文本分类任务压缩包内未检测到任何CSV数据文件。'}
        
        # 检查是否有其他非CSV数据文件
        if len(other_non_system_files) > 0:
            return {'status': 'error', 'message': f'文本分类任务压缩包中检测到非CSV数据文件: "{other_non_system_files[0]}"。请确保压缩包只包含CSV文件。'}
        
        return {'status': 'success', 'message': '数据文件内容符合文本分类任务要求。'}

    else:
        return {'status': 'error', 'message': '不支持的任务类型。'}
    

@csrf_exempt 
@require_POST
def validate_data_file(request):
    """
    接收数据文件（压缩包）和任务类型，进行内容预检查。
    """
    if 'data_file' not in request.FILES:
        error_msg = '未检测到数据文件。'
        logger.error(f"validate_data_file: {error_msg}")
        return JsonResponse({'status': 'error', 'message': error_msg}, status=400)

    data_file = request.FILES['data_file']
    task_type = request.POST.get('task_type')

    if not task_type:
        error_msg = f'任务类型 (task_type) 为空，无法进行文件内容校验。文件: {data_file.name}'
        logger.error(f"validate_data_file: {error_msg}")
        return JsonResponse({'status': 'error', 'message': error_msg}, status=400)

    # 检查文件大小
    if data_file.size > 2 * 1024 * 1024 * 1024:
        error_msg = f'数据文件不能超过 2GB。文件: {data_file.name}, 大小: {data_file.size}'
        logger.warning(f"validate_data_file: {error_msg}")
        return JsonResponse({'status': 'error', 'message': error_msg}, status=400)

    file_extension = os.path.splitext(data_file.name)[1].lower()
    temp_file_path = None

    try:
        # 优化大文件读取，避免 data_file.read() 将整个文件读入内存
        if file_extension == '.zip':
            with zipfile.ZipFile(data_file.file, 'r') as zf: # 直接使用 data_file.file
                file_list = zf.namelist()
                result = _check_compressed_file_contents(file_list, task_type)
                if result['status'] == 'error':
                    logger.warning(f"validate_data_file (zip content error): {result['message']}")
                return JsonResponse(result, status=200 if result['status'] == 'success' else 400)
        elif file_extension == '.rar':
            with rarfile.RarFile(data_file.file, 'r') as rf: # 直接使用 data_file.file
                file_list = [f.filename for f in rf.infolist()]
                result = _check_compressed_file_contents(file_list, task_type)
                if result['status'] == 'error':
                    logger.warning(f"validate_data_file (rar content error): {result['message']}")
                return JsonResponse(result, status=200 if result['status'] == 'success' else 400)
        elif file_extension == '.7z':
            temp_upload_dir = Path(settings.MEDIA_ROOT) / 'temp_upload'
            temp_upload_dir.mkdir(parents=True, exist_ok=True)
            
            temp_file_path = temp_upload_dir / f"{uuid.uuid4().hex}_{data_file.name}"
            
            with open(temp_file_path, 'wb+') as temp_f:
                for chunk in data_file.chunks():
                    temp_f.write(chunk)
            
            with py7zr.SevenZipFile(str(temp_file_path), mode='r') as szf:
                file_list = szf.getnames()
                result = _check_compressed_file_contents(file_list, task_type)
                if result['status'] == 'error':
                    logger.warning(f"validate_data_file (7z content error): {result['message']}")
                return JsonResponse(result, status=200 if result['status'] == 'success' else 400)
        else:
            error_msg = f'不支持的压缩文件格式: {file_extension}。请上传.zip, .rar 或 .7z 格式文件。文件: {data_file.name}'
            logger.warning(f"validate_data_file: {error_msg}")
            return JsonResponse({'status': 'error', 'message': error_msg}, status=400)

    except (zipfile.BadZipFile, rarfile.BadRarFile, py7zr.Bad7zFile) as e:
        error_msg = f'上传的文件是损坏的压缩包或无法识别的格式。请检查文件。文件: {data_file.name}, 错误: {e}'
        logger.warning(f"validate_data_file: {error_msg}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': error_msg}, status=400)
    except rarfile.MissingUnrarError:
        error_msg = '服务器缺少处理RAR文件的必要组件（unrar）。请联系管理员。'
        logger.error(f"validate_data_file: {error_msg}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': error_msg}, status=500)
    except Exception as e:
        error_msg = f'服务器内部错误，处理文件时发生意外：{e}。文件: {data_file.name}'
        logger.exception(f"validate_data_file: {error_msg}") # 使用logger.exception记录完整堆栈
        return JsonResponse({'status': 'error', 'message': error_msg}, status=500)
    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logger.debug(f"validate_data_file: Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"validate_data_file: Failed to remove temporary file {temp_file_path}: {e}", exc_info=True)


# 渲染首页中当前用户的任务统计数据
@login_required
@require_http_methods(["GET"]) # 确保只接受 GET 请求
def get_homepage_stats(request):
    """
    提供首页所需的最新任务统计数据 (进行中、已完成)。
    """
    try:
        # 计算当前用户的“进行中”任务数量 (包括 'processing' 和 'pending')
        in_progress_count = Task.objects.filter(
            user=request.user,
            status__in=['processing', 'pending']
        ).count()

        # 计算当前用户的“已完成”任务数量 (只包含 'completed')
        completed_count = Task.objects.filter(
            user=request.user,
            status='completed'
        ).count()

        return JsonResponse({
            'status': 'success',
            'in_progress_count': in_progress_count,
            'completed_count': completed_count
        })
    except Exception as e:
        logger.error(f"获取首页统计数据时出错: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': '服务器内部错误'}, status=500)


# 获取草稿/暂存数据
@require_http_methods(["GET"])
def get_draft(request):
    draft = request.session.get('task_draft', {})
    return JsonResponse(draft)


# 保存草稿数据
@csrf_exempt
@require_http_methods(["POST"])
def save_draft(request):
    try:
        request.session['task_draft'] = {
            'name': request.POST.get('task_name'),
            'description': request.POST.get('task_desc'),
            'task_type': request.POST.get('task_type')
        }
        return JsonResponse({'status': 'success'})
    except Exception as e:
        logger.error(f"保存草稿失败: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# 提交任务
@csrf_exempt
@login_required
@require_http_methods(["POST"])
def submit_task(request):
    submission_lock_key = None
    lock_acquired = False

    try:
        # 用户登录检查
        if not request.user.is_authenticated:
            raise PermissionDenied("用户未登录")

        # 获取提交数据
        task_name = request.POST.get('task_name', '').strip()
        task_type = request.POST.get('task_type', '')
        task_desc = request.POST.get('task_desc', '')
        raw_labeling_type = request.POST.get('labeling_type', '').strip()
        data_file = request.FILES.get('data_file')
        label_file = request.FILES.get('label_file')
        model_choice = request.POST.get('model_choice')
        model_strength = request.POST.get('model_strength')

        # 兼容前端不再传 labeling_type 的场景：有标注文件视为半监督，否则默认预训练。
        if raw_labeling_type in ('pre-trained', 'semi-supervised'):
            labeling_type = raw_labeling_type
        else:
            labeling_type = 'semi-supervised' if label_file else 'pre-trained'
            logger.warning(
                "submit_task 未收到有效 labeling_type，已自动推断为 %s (raw=%r)",
                labeling_type,
                raw_labeling_type,
            )

        if not task_name or not task_type or not data_file:
            return JsonResponse({'status': 'error', 'message': '任务名称、任务类型和数据文件为必填项'}, status=400)

        # 服务端幂等防重：同一用户短时间内的同内容提交只创建一个任务
        fingerprint_raw = '|'.join([
            str(request.user.id),
            task_name,
            task_type,
            task_desc,
            labeling_type,
            str(model_choice or ''),
            str(model_strength or ''),
            str(getattr(data_file, 'name', '')),
            str(getattr(data_file, 'size', 0)),
            str(getattr(label_file, 'name', '')),
            str(getattr(label_file, 'size', 0)),
        ])
        fingerprint = hashlib.sha256(fingerprint_raw.encode('utf-8')).hexdigest()
        submission_lock_key = f"submit_task_lock:{fingerprint}"
        lock_acquired = cache.add(submission_lock_key, 'IN_PROGRESS', timeout=30)

        if not lock_acquired:
            cached_task_id = cache.get(submission_lock_key)
            if cached_task_id and cached_task_id != 'IN_PROGRESS':
                existing_task = Task.objects.filter(id=cached_task_id, user=request.user).first()
                if existing_task:
                    return JsonResponse({
                        'status': 'success',
                        'task_id': existing_task.id,
                        'task_type': existing_task.task_type,
                        'message': '检测到重复提交，已返回已创建任务',
                        'monitor_url': f'/api/tasks/{existing_task.id}/'
                    })
            return JsonResponse({'status': 'error', 'message': '任务正在提交中，请勿重复操作'}, status=429)

        # 消息中间件连接检查
        try:
            with celery_app.connection() as conn:
                conn.ensure_connection(max_retries=3, timeout=10)
        except OperationalError as e:
            logger.error(f"消息中间件连接异常: {str(e)}", exc_info=True)
            if lock_acquired and submission_lock_key:
                cache.delete(submission_lock_key)
            return JsonResponse(
                {'status': 'error', 'message': '后台服务初始化中，请15秒后重试'},
                status=503
            )

        # 创建任务记录
        task = Task.objects.create(
            user=request.user,
            name=task_name,
            description=task_desc,
            task_type=task_type,
            classification_scene='topic' if task_type == 'text-classification' else None,
            labeling_type=labeling_type,
            data_file=data_file,
            label_file=label_file,
            status='pending',
            model_choice=model_choice,
            model_strength=model_strength,
            label_list=[],
            progress=0.0,  # 初始化进度字段
        )

        # 若是文本分类，预置标签为 正面/负面/中性（已不再区分分类场景，所有文本分类都使用主题分类模式）
        # 标签配置由用户上传的标注文件决定

        # 将上传文件进行解压并处理异常
        try:
            task_id = str(task.id)
            extract_dir = Path(settings.MEDIA_ROOT) / 'extracted' / task_id
            extract_dir = extract_dir.absolute().resolve()
            try:
                extract_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                return JsonResponse({'status': 'error', 'message': f'目录权限不足: {str(e)}'}, status=500)

            file_extension = Path(getattr(data_file, 'name', '')).suffix.lower()

            # 使用统一的解压策略，和 validate_data_file 支持格式保持一致
            if file_extension == '.zip':
                with zipfile.ZipFile(data_file.file, 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        if file.startswith(('__MACOSX/', '.DS_Store')):
                            continue
                        zip_ref.extract(file, extract_dir)
            elif file_extension == '.rar':
                with rarfile.RarFile(data_file.file) as rar_ref:
                    for file in rar_ref.namelist():
                        if file.startswith(('__MACOSX/', '.DS_Store')):
                            continue
                        rar_ref.extract(file, path=extract_dir)
            elif file_extension == '.7z':
                with py7zr.SevenZipFile(data_file.file, mode='r') as szf:
                    members = [
                        m for m in szf.getnames()
                        if not m.startswith(('__MACOSX/', '.DS_Store'))
                    ]
                    if members:
                        szf.extract(path=extract_dir, targets=members)
            else:
                raise ValueError(f'不支持的压缩格式: {file_extension}')

            # 将解压路径保存到 Task 中
            task.extracted_dir = str(extract_dir)
            # 先保存解压路径
            task.save(update_fields=['extracted_dir'])

            # 统计上传的总数并保存为固定值，后续界面显示该固定数值不随标注变更而改变
            total_count = 0
            try:
                if task.task_type == 'image-classification':
                    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
                    total_count = sum(1 for p in extract_dir.rglob('*') if p.suffix.lower() in exts)
                elif task.task_type == 'text-classification':
                    total_count = 0
                    for p in extract_dir.rglob('*.csv'):
                        try:
                            with open(p, 'r', encoding='utf-8', newline='') as f:
                                reader = csv.reader(f)
                                rows = list(reader)
                                if len(rows) > 1:
                                    total_count += max(0, len(rows) - 1)
                        except Exception:
                            continue
                else:
                    total_count = sum(1 for p in extract_dir.rglob('*') if p.is_file() and not p.name.startswith('.'))
            except Exception:
                total_count = 0

            task.uploaded_total = total_count
            task.save(update_fields=['uploaded_total'])
        except OSError as e:
            if lock_acquired and submission_lock_key:
                cache.delete(submission_lock_key)
            return JsonResponse({'status': 'error', 'message': f'路径错误: {str(e)}'}, status=500)
        except (zipfile.BadZipFile, rarfile.BadRarFile, py7zr.Bad7zFile):
            if lock_acquired and submission_lock_key:
                cache.delete(submission_lock_key)
            return JsonResponse({'status': 'error', 'message': '文件不是有效的压缩包格式'}, status=400)
        except rarfile.MissingUnrarError:
            if lock_acquired and submission_lock_key:
                cache.delete(submission_lock_key)
            return JsonResponse({'status': 'error', 'message': '服务器缺少 RAR 解压组件（unrar）'}, status=500)
        except ValueError as e:
            if lock_acquired and submission_lock_key:
                cache.delete(submission_lock_key)
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

        # 半监督情境下存储标注文件
        if labeling_type == 'semi-supervised' and label_file:
            label_dir = Path(settings.MEDIA_ROOT) / 'extracted' / (str(task_id) + '_label')
            label_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

            # 清理文件名
            import re
            label_file_name = re.sub(r'[^\w\s.-]', '_', label_file.name)
            label_file_path = label_dir / label_file_name

            try:
                with open(label_file_path, "wb+") as destination:
                    for chunk in label_file.chunks():
                        destination.write(chunk)
                print(f"File saved successfully at {label_file_path}") # 调试信息
            except Exception as e:
                print(f"Error saving file: {e}") # 调试信息

        # 启动Celery任务时传递任务类型
        try:
            celery_task = None
            if labeling_type == 'pre-trained':
                # 预训练模式下使用process_pretrained_task
                celery_task = process_pretrained_task.apply_async(
                    args=[task.id],
                    task_id=f'pretrained_{task.id}_{uuid.uuid4().hex[:6]}'
                )
            elif labeling_type == 'semi-supervised':
                # 半监督模式下使用原有的process_annotation_task
                celery_task = process_annotation_task.apply_async(
                    args=[task.id],
                    task_id=f'semisupervised_{task.id}_{uuid.uuid4().hex[:6]}'
                )
            else:
                # 理论上不会进入此分支，保留为兜底，避免任务记录创建后未入队。
                logger.warning("未知 labeling_type=%s，回退到预训练任务", labeling_type)
                celery_task = process_pretrained_task.apply_async(
                    args=[task.id],
                    task_id=f'pretrained_{task.id}_{uuid.uuid4().hex[:6]}'
                )

            task.celery_task_id = celery_task.id
            task.save(update_fields=['celery_task_id'])
            if lock_acquired and submission_lock_key:
                cache.set(submission_lock_key, task.id, timeout=30)
        except Exception as e:
            task.status = 'failed'
            task.error_log = str(e)[:4096]
            task.save()
            if lock_acquired and submission_lock_key:
                cache.delete(submission_lock_key)
            raise

        return JsonResponse({
            'status': 'success',
            'task_id': task.id,
            'task_type': task.task_type,  # 返回任务类型
            'message': '任务已提交，正在处理中',
            'monitor_url': f'/api/tasks/{task.id}/'  # 添加监控端点
        })

    except Exception as e:
        # 记录详细错误日志
        logger.error(f"提交任务时出错: {e}", exc_info=True)
        if lock_acquired and submission_lock_key and cache.get(submission_lock_key) == 'IN_PROGRESS':
            cache.delete(submission_lock_key)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# 清除提交表单数据
@csrf_exempt
@require_http_methods(["POST"])
def clear_draft(request):
    try:
        if 'task_draft' in request.session:
            del request.session['task_draft']
        return JsonResponse({'status': 'success'})
    except Exception as e:
        logger.error(f"清除草稿失败: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@login_required
@require_http_methods(["GET"])
def get_task_meta_data(request, task_id):
    """
    获取指定任务的所有可能的标签和置信度值，用于前端筛选器初始化。
    """
    try:
        task = get_object_or_404(Task, id=task_id)
        TaskService.validate_ownership(task, request.user)

        config = TASK_CONFIG[task.task_type]
        result_model = config['result_model']
        relation_field = config['relation_field']

        # 获取所有唯一的标签
        # 排除 None 或空字符串的标签
        labels = list(result_model.objects.filter(
            **{relation_field: task}
        ).exclude(label__isnull=True).exclude(label__exact='')
        .values_list('label', flat=True).distinct().order_by('label'))

        # 获取所有唯一的置信度
        confidences_raw = list(result_model.objects.filter(
            **{relation_field: task}
        ).exclude(confidence__isnull=True)
        .values_list('confidence', flat=True).distinct())

        # 格式化置信度为3位小数的字符串，并特殊处理 -1
        # 使用 set 确保唯一性，然后排序
        confidences = sorted(list(set([
            f"{float(c):.3f}" if c is not None and float(c) != -1.0 else "-1.000"
            for c in confidences_raw
        ])), key=lambda x: float(x)) # 按数值排序

        # 状态选项固定为这三种，前端直接使用
        # 确保statuses列表顺序与前端STATUS_MAP_CN_TO_EN定义的顺序一致，以保证默认排序行为
        statuses = ['已校验', '未校验', '已标注']

        return JsonResponse({'status': 'success', 'labels': labels, 'confidences': confidences, 'statuses': statuses})
    except Task.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '任务不存在'}, status=404)
    except Exception as e:
        logger.error(f"获取任务元数据时出错: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'服务器内部错误: {str(e)}'}, status=500)


# 获取任务列表
@login_required
def get_tasks(request):
    if request.method == 'GET':
        # 获取筛选参数
        status_list = request.GET.getlist('status', [])
        task_type = request.GET.get('task_type', '')
        task_name = request.GET.get('name', '')

        # 映射“进行中”和“已完成”状态
        mapped_status = []
        for status in status_list:
            if status == 'processing':
                mapped_status.extend(['processing', 'pending'])
            elif status == 'completed':
                mapped_status.extend(['completed', 'checking'])
            else:
                mapped_status.append(status)
        mapped_status = list(set(mapped_status))

        # 构建查询条件
        query = Q(user=request.user)

        if status_list:
            query &= Q(status__in=mapped_status)
        if task_type:
            query &= Q(task_type=task_type)
        if task_name:
            query &= Q(name__icontains=task_name)

        # 标注是否收藏
        tasks_queryset = Task.objects.filter(query).annotate(
            is_favorite=Exists(
                FavoriteTask.objects.filter(
                    user=request.user,
                    task=OuterRef('pk')
                )
            )
        )

        # 优先按照收藏状态降序，然后按更新时间降序
        tasks_queryset = tasks_queryset.order_by('-is_favorite', '-updated_at').values(
            'id',
            'name',
            'task_type',
            'status',
            'progress',
            'description',
            'created_at',
            'updated_at',
            'is_favorite'  # 添加收藏状态字段
        )

        # 分页参数
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 12))

        # 分页查询
        paginator = Paginator(tasks_queryset, page_size)
        page_obj = paginator.get_page(page)

        return JsonResponse({
            'results': list(page_obj.object_list),
            'total': paginator.count
        }, safe=False)
    return JsonResponse({'error': 'Invalid method'}, status=405)


def _manual_label_manifest_path(task_id):
    return Path(settings.MEDIA_ROOT) / 'results' / str(task_id) / 'manual_labels.csv'


def _merge_manual_labels_for_task(task):
    """将原始标注文件与人工补标文件合并，人工补标优先。"""
    if not task.label_file or not Path(task.label_file.path).exists():
        raise FileNotFoundError("原始标注文件不存在，无法合并人工标注。")

    manual_label_path = _manual_label_manifest_path(task.id)
    if not manual_label_path.exists():
        return Path(task.label_file.path)

    merged_label_path = Path(settings.MEDIA_ROOT) / 'results' / str(task.id) / 'merged_manual_labels.csv'
    merged_label_path.parent.mkdir(parents=True, exist_ok=True)

    merged_records = {}

    for source_path in [Path(task.label_file.path), manual_label_path]:
        with open(source_path, 'r', encoding='utf-8', newline='') as source_file:
            reader = csv.DictReader(source_file)
            for row in reader:
                filename = (row.get('filename') or '').strip()
                label = (row.get('class') or '').strip()
                if filename:
                    merged_records[filename] = label

    with open(merged_label_path, 'w', encoding='utf-8', newline='') as target_file:
        writer = csv.writer(target_file)
        writer.writerow(['filename', 'class'])
        for filename, label in sorted(merged_records.items()):
            writer.writerow([filename, label])

    return merged_label_path


def _upsert_manual_label_record(task, image_name, label):
    manifest_path = _manual_label_manifest_path(task.id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records = {}
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8', newline='') as source_file:
            reader = csv.DictReader(source_file)
            for row in reader:
                filename = (row.get('filename') or '').strip()
                existing_label = (row.get('class') or '').strip()
                if filename:
                    records[filename] = existing_label

    records[image_name] = label

    with open(manifest_path, 'w', encoding='utf-8', newline='') as target_file:
        writer = csv.writer(target_file)
        writer.writerow(['filename', 'class'])
        for filename, existing_label in sorted(records.items()):
            writer.writerow([filename, existing_label])

    return manifest_path


def _load_image_uncertainty_scores(task):
    """读取图像分类结果文件中的不确定性分数映射。"""
    result_path = Path(settings.MEDIA_ROOT) / 'results' / str(task.id) / 'result.csv'
    if not result_path.exists():
        return {}

    uncertainty_scores = {}

    try:
        with open(result_path, 'r', encoding='utf-8', newline='') as source_file:
            reader = csv.DictReader(source_file)
            for row in reader:
                image_name = (row.get('image_name') or row.get('image') or '').strip()
                if not image_name:
                    continue

                score = None
                for key in ('uncertainty_score', 'entropy', 'margin'):
                    raw_value = row.get(key)
                    if raw_value not in (None, ''):
                        try:
                            score = float(raw_value)
                            if key == 'margin':
                                score = 1.0 - score
                            break
                        except ValueError:
                            continue

                if score is None:
                    confidence_raw = row.get('confidence')
                    try:
                        confidence_value = float(confidence_raw) if confidence_raw not in (None, '') else 0.0
                        score = max(0.0, 1.0 - confidence_value)
                    except ValueError:
                        score = 0.0

                uncertainty_scores[image_name] = score
    except Exception as exc:
        logger.warning("读取任务 %s 的不确定性分数失败: %s", task.id, exc, exc_info=True)

    return uncertainty_scores


def _load_text_uncertainty_scores(task):
    """读取文本分类结果文件中的不确定性分数映射。"""
    result_path = Path(settings.MEDIA_ROOT) / 'results' / str(task.id) / 'result.csv'
    if not result_path.exists():
        return {}

    uncertainty_scores = {}

    try:
        with open(result_path, 'r', encoding='utf-8', newline='') as source_file:
            reader = csv.DictReader(source_file)
            for row in reader:
                text_id = (row.get('text_id') or '').strip()
                if not text_id:
                    continue

                score = None
                for key in ('uncertainty_score', 'entropy', 'margin'):
                    raw_value = row.get(key)
                    if raw_value not in (None, ''):
                        try:
                            score = float(raw_value)
                            if key == 'margin':
                                score = 1.0 - score
                            break
                        except ValueError:
                            continue

                if score is None:
                    confidence_raw = row.get('confidence')
                    try:
                        confidence_value = float(confidence_raw) if confidence_raw not in (None, '') else 0.0
                        score = max(0.0, 1.0 - confidence_value)
                    except ValueError:
                        score = 0.0

                uncertainty_scores[text_id] = score
    except Exception as exc:
        logger.warning("读取任务 %s 的文本不确定性分数失败: %s", task.id, exc, exc_info=True)

    return uncertainty_scores


# 对任务进行获取或删除操作
@login_required
def task_operation(request, task_id):
    task = get_object_or_404(Task, id=task_id, user=request.user)

    if request.method == 'GET':
        return JsonResponse({
            'id': task.id,
            'name': task.name, # 返回名称
            'status': task.status,
            'progress': task.progress,
            'created_at': task.created_at.strftime('%Y-%m-%d %H:%M'),
            'task_type': task.get_task_type_display(),
            'description': task.description # 返回描述
        })

    elif request.method == 'DELETE':
        # 终止关联的 Celery 任务，并尽量清除 broker/结果后端中的残留。
        if task.celery_task_id:
            from smartlabel.celery import app
            try:
                # terminate=True 会尝试终止正在执行的子进程；
                # 同时会将任务加入 revoked 集合，避免后续再次执行。
                app.control.revoke(task.celery_task_id, terminate=True, signal='SIGTERM')
                logger.info(f"已请求撤销Celery任务: {task.celery_task_id}")
            except Exception as e:
                logger.warning(f"撤销Celery任务 {task.celery_task_id} 失败: {e}")

            try:
                app.AsyncResult(task.celery_task_id).forget()
                logger.info(f"已清理Celery结果后端记录: {task.celery_task_id}")
            except Exception as e:
                logger.warning(f"清理Celery结果后端失败 {task.celery_task_id}: {e}")

            removed = _remove_task_message_from_broker(task.celery_task_id)
            if removed:
                logger.info("已从 Redis broker 队列移除 %s 条残留消息(task_id=%s)", removed, task.celery_task_id)

        # 删除上传的原始文件
        if task.data_file:
            if os.path.exists(task.data_file.path):
                os.remove(task.data_file.path)
                logger.info(f"已删除数据文件: {task.data_file.path}")

        if task.label_file:
            if os.path.exists(task.label_file.path):
                os.remove(task.label_file.path)
                logger.info(f"已删除标签文件: {task.label_file.path}")

        # 清理解压目录
        extracted_dir = Path(settings.MEDIA_ROOT) / 'extracted' / str(task_id)
        if extracted_dir.exists():
            import shutil
            shutil.rmtree(extracted_dir)
            logger.info(f"已删除解压目录: {extracted_dir}")

        # 清除标签目录
        label_dir = Path(settings.MEDIA_ROOT) / 'extracted' / (str(task_id) + '_label')
        if label_dir.exists():
            import shutil
            shutil.rmtree(label_dir)
            logger.info(f"已删除标签目录: {label_dir}")

        # 删除结果文件
        results_dir = Path(settings.MEDIA_ROOT) / 'results' / str(task_id)
        if results_dir.exists():
            import shutil
            shutil.rmtree(results_dir)
            logger.info(f"已删除结果目录: {results_dir}")

        # 删除任务记录及相关结果 (结果记录会在 Task.delete() 时通过 CASCADE 自动删除)
        task.delete()
        logger.info(f"已删除任务记录: {task_id}")

        return JsonResponse({'status': 'success'})

    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
def manual_annotation(request, task_id):
    """主动学习人工标注界面。"""
    task = get_object_or_404(Task, id=task_id, user=request.user)

    if task.task_type not in ('image-classification', 'text-classification'):
        return JsonResponse({'status': 'error', 'message': '当前页面仅支持图像分类或文本分类任务'}, status=400)

    config = TASK_CONFIG[task.task_type]
    result_model = config['result_model']
    relation_field = config['relation_field']

    # 统一口径：未标注（未校验）计数需排除已被人工标注（confidence=-1）的样本
    unlabeled_count = result_model.objects.filter(**{relation_field: task}, status='unverified').exclude(confidence=-1).count()

    context = TaskService.get_common_context(task)
    context.update({
        'task_view_mode': 'manual',
        'task_page_title': '主动学习标注',
        'unlabeled_count': unlabeled_count,
    })

    template_name = 'platform/tasks/manual_annotation_text_h.html' if task.task_type == 'text-classification' else 'platform/tasks/manual_annotation_h.html'
    return render(request, template_name, context)


@login_required
def unlabeled_detail(request, task_id):
    """未标注样本界面（模型预测结果）。"""
    task = get_object_or_404(Task, id=task_id, user=request.user)

    # 文本分类不支持未标注界面，重定向到已标注界面
    if task.task_type == 'text-classification':
        return redirect('task_detail', task_id=task_id)

    if task.task_type not in ('image-classification',):
        return JsonResponse({'status': 'error', 'message': '当前页面仅支持图像分类任务'}, status=400)

    config = TASK_CONFIG[task.task_type]
    result_model = config['result_model']
    relation_field = config['relation_field']

    # 统一口径：未标注（未校验）计数需排除已被人工标注（confidence=-1）的样本
    unlabeled_count = result_model.objects.filter(**{relation_field: task}, status='unverified').exclude(confidence=-1).count()

    context = TaskService.get_common_context(task)
    context.update({
        'task_view_mode': 'unlabeled',
        'task_page_title': '未标注样本',
        'unlabeled_count': unlabeled_count,
    })

    template_name = 'platform/tasks/unlabeled_detail_h.html'
    return render(request, template_name, context)


# 获取任务进度
@require_http_methods(["GET"])
def task_progress(request, task_id):
    """获取任务状态和进度"""
    try:
        task = Task.objects.get(id=task_id)
        heartbeat = cache.get(f"task:{task_id}:algo_heartbeat") or {}
        return JsonResponse({
            'status': task.status,
            'progress': task.progress,
            'error': task.error_log if task.status == 'failed' else None,
            'algorithm_last_log_at': heartbeat.get('at'),
            'algorithm_last_log': heartbeat.get('line'),
        })
    except Task.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '任务不存在'}, status=404)
    except Exception as e:
        logger.error(f"获取任务进度时出错: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@login_required
def task_detail(request, task_id):
    """统一详情视图"""
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user)

    # 检查收藏状态
    is_favorite = FavoriteTask.objects.filter(
        user=request.user,
        task=task
    ).exists()

    config = TASK_CONFIG[task.task_type]
    result_model = config['result_model']
    relation_field = config['relation_field']
    total_count = result_model.objects.filter(**{relation_field: task}).count()

    stats = _get_label_verification_stats(task, result_model, relation_field)

    # 计算校验进度
    verification_progress = 0
    total_verifiable = stats.get('total_verifiable_count', total_count)
    verified = stats.get('verified_count', 0)
    if total_verifiable > 0:
        verification_progress = round((verified / total_verifiable) * 100, 1)

    # 附加统计数据到 task 对象（用于模板渲染）
    task.total_verifiable_count = total_verifiable
    task.verified_count = verified
    task.unverified_count = stats.get('unverified_count', 0)
    task.manual_annotated_count = stats.get('manual_annotated_count', stats.get('total_labeled_count', 0)) # 新增：人工标注数量（即已标注总数）
    task.verification_progress = verification_progress # 新增：校验进度
    task.total_count = total_count

    # 回退逻辑：如果数据库中尚无样本（例如刚提交），尝试根据解压目录估算总数，便于页面初次渲染显示正确的上传数量
    if task.total_count == 0 and getattr(task, 'extracted_dir', None):
        try:
            extract_path = Path(task.extracted_dir)
            if extract_path.exists():
                if task.task_type == 'image-classification':
                    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
                    task.total_count = sum(1 for p in extract_path.rglob('*') if p.suffix.lower() in exts)
                elif task.task_type == 'text-classification':
                    cnt = 0
                    for p in extract_path.rglob('*.csv'):
                        try:
                            with open(p, 'r', encoding='utf-8', newline='') as f:
                                reader = csv.reader(f)
                                rows = list(reader)
                                if len(rows) > 1:
                                    cnt += max(0, len(rows) - 1)
                        except Exception:
                            continue
                    task.total_count = cnt
                else:
                    task.total_count = sum(1 for p in extract_path.rglob('*') if p.is_file() and not p.name.startswith('.'))
        except Exception:
            task.total_count = result_model.objects.filter(**{relation_field: task}).count()

    # 获取所有条目，用于页面渲染和后续处理 (不需要一次性加载所有数据，这里只是为了获取所有可能的标签)
    items = result_model.objects.filter(**{relation_field: task})

    context = TaskService.get_common_context(task)
    if task.task_type in ('image-classification', 'text-classification') and task.labeling_type == 'pre-trained':
        context['task_completion_redirect_url'] = reverse('manual_annotation', args=[task.id])
    context.update({
        'items': items,
        'is_favorite': is_favorite,  # 添加收藏状态
        'task_view_mode': 'labeled',
        'task_page_title': '已标注',
        # 增加 if item.label 避免标签为None时出错
        'labels_json': json.dumps(list(
            set(item.label for item in items if item.label)
        ))
    })

    return render(request, config['detail_template'], context)


@csrf_exempt
@require_http_methods(["POST"])
def handle_label_update(request, task_id):
    """统一标签更新"""
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user)

    config = TASK_CONFIG[task.task_type]
    model = config['result_model']
    item_id = request.POST.get('item_id')
    new_label = request.POST.get('label')

    if not item_id or new_label is None: # new_label可以是空字符串
        return JsonResponse(
            {'status': 'error', 'message': '缺少必要参数'},
            status=400
        )

    try:
        item = model.objects.get(id=item_id)
        # 如果标签未修改且当前已经是 'verified' 状态，则返回 'no_change'
        if item.label == new_label and item.status == 'verified':
             return JsonResponse({'status': 'no_change', 'message': '标签未修改，且已是校验状态'})

        item.label = new_label
        # 标签修改后回到“未校验”，需要用户显式点击“通过”完成校验。
        item.status = 'unverified'
        # 如果是人工标注的，其 confidence 应该保持 -1.0。这里不改变 confidence。
        item.save()
        return JsonResponse({'status': 'success', 'message': '标签更新成功'})

    except model.DoesNotExist:
        return JsonResponse(
            {'status': 'error', 'message': '记录不存在'},
            status=404
        )
    except Exception as e:
        logger.error(f"更新标签失败 (task_id: {task_id}, item_id: {item_id}): {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'服务器内部错误: {str(e)}'}, status=500)


@login_required
@require_POST
def save_manual_annotation(request, task_id):
    """保存主动学习界面的人工标注结果。"""
    task = get_object_or_404(Task, id=task_id, user=request.user)

    if task.task_type not in ('image-classification', 'text-classification'):
        return JsonResponse({'status': 'error', 'message': '当前页面仅支持图像分类或文本分类任务'}, status=400)

    item_id = request.POST.get('item_id')
    label = request.POST.get('label', '').strip()

    if not item_id or not label:
        return JsonResponse({'status': 'error', 'message': '缺少必要参数'}, status=400)

    try:
        model = TASK_CONFIG[task.task_type]['result_model']
        item = model.objects.get(id=item_id, task=task)
        item.label = label
        item.status = 'unverified'
        item.confidence = -1.0
        item.save(update_fields=['label', 'status', 'confidence'])

        if task.task_type == 'image-classification':
            _upsert_manual_label_record(task, item.image_name, label)

        return JsonResponse({
            'status': 'success',
            'message': '人工标注已保存',
        })
    except model.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '记录不存在'}, status=404)
    except Exception as e:
        logger.error(f"保存人工标注失败 (task_id: {task_id}, item_id: {item_id}): {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'服务器内部错误: {str(e)}'}, status=500)


@login_required
@require_POST
def add_unlabeled_to_labeled(request, task_id):
    """将未标注样本加入已标注界面（使用预测标签或用户修正标签）。"""
    task = get_object_or_404(Task, id=task_id, user=request.user)

    if task.task_type not in ('image-classification', 'text-classification'):
        return JsonResponse({'status': 'error', 'message': '当前页面仅支持图像分类或文本分类任务'}, status=400)

    item_id = request.POST.get('item_id')
    label = request.POST.get('label', '').strip()

    if not item_id:
        return JsonResponse({'status': 'error', 'message': '缺少必要参数 item_id'}, status=400)

    try:
        model = TASK_CONFIG[task.task_type]['result_model']
        item = model.objects.get(id=item_id, task=task)

        if not label:
            label = (item.label or '').strip()

        if not label:
            return JsonResponse({'status': 'error', 'message': '标签不能为空'}, status=400)

        item.label = label
        item.confidence = -1.0
        item.status = 'unverified'
        item.save(update_fields=['label', 'confidence', 'status'])

        if task.task_type == 'image-classification':
            _upsert_manual_label_record(task, item.image_name, label)

        return JsonResponse({'status': 'success', 'message': '已加入已标注界面'})
    except model.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '记录不存在'}, status=404)
    except Exception as e:
        logger.error(f"未标注样本加入已标注失败 (task_id: {task_id}, item_id: {item_id}): {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'服务器内部错误: {str(e)}'}, status=500)


@login_required
@require_POST
def rerun_semi_supervised(request, task_id):
    """在人工标注后重新调用半监督流程。"""
    task = get_object_or_404(Task, id=task_id, user=request.user)

    if task.task_type not in ('image-classification', 'text-classification'):
        return JsonResponse({'status': 'error', 'message': '当前页面仅支持图像分类或文本分类任务'}, status=400)

    try:
        if task.celery_task_id:
            app = celery_app
            try:
                app.control.revoke(task.celery_task_id, terminate=True, signal='SIGTERM')
            except Exception:
                pass

        task.status = 'pending'
        task.progress = 0
        task.error_log = ''
        task.save(update_fields=['status', 'progress', 'error_log'])

        celery_task = process_annotation_task.apply_async(
            args=[task.id],
            task_id=f'manual_retrain_{task.id}_{uuid.uuid4().hex[:6]}'
        )
        task.celery_task_id = celery_task.id
        task.save(update_fields=['celery_task_id'])

        return JsonResponse({
            'status': 'success',
            'message': '已重新调用半监督算法',
            'task_id': task.id,
        })
    except Exception as e:
        logger.error(f"重新调用半监督算法失败 (task_id: {task_id}): {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'服务器内部错误: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def update_task_items(request, task_id):
    """
    根据前端传递的筛选条件，批量更新所有匹配项的状态。
    这个视图的URL应该是 /tasks/<task_id>/updateall/
    """
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user)

    config = TASK_CONFIG[task.task_type]
    model = config['result_model']
    relation_field = config['relation_field']

    try:
        # 前端传过来的 status_to_set 应该是英文 'verified' 或 'unverified'
        status_to_set_en = request.POST.get('status_to_set')
        if status_to_set_en not in ['verified', 'unverified']:
            return JsonResponse({'status': 'error', 'message': '无效的目标状态值'}, status=400)

        # 解码URL参数
        labels_str = urllib.parse.unquote(request.POST.get('label', ''))
        confidences_str = urllib.parse.unquote(request.POST.get('confidence', ''))
        statuses_raw_cn = urllib.parse.unquote(request.POST.get('status', '')) # 前端传递的中文状态

        queryset = model.objects.filter(**{relation_field: task})

        final_q_conditions = [] # 用于 AND 连接的 Q 对象列表

        # 标签筛选
        if labels_str:
            labels = [l.strip() for l in labels_str.split(',') if l.strip()]
            if labels:
                final_q_conditions.append(Q(label__in=labels))

        # 置信度筛选
        if confidences_str:
            confidences = [c.strip() for c in confidences_str.split(',') if c.strip()]
            numeric_confidences = []
            for c_str in confidences:
                try:
                    # 将前端的 "-1.000" 映射回数据库的 -1.0
                    if c_str == '-1.000':
                        numeric_confidences.append(-1.0)
                    else:
                        numeric_confidences.append(float(c_str))
                except ValueError:
                    logger.warning(f"跳过无效置信度筛选值: {c_str}")
                    pass
            if numeric_confidences:
                final_q_conditions.append(Q(confidence__in=numeric_confidences))

        # 状态筛选逻辑 (前端中文 -> 后端数据库逻辑)
        if statuses_raw_cn:
            statuses_cn = [s.strip() for s in statuses_raw_cn.split(',') if s.strip()]

            current_status_q_objects = []
            for status_cn in statuses_cn:
                if status_cn == '已标注':
                    # '已标注' 对应 confidence = -1
                    current_status_q_objects.append(Q(confidence=-1))
                elif status_cn in ['已校验', '未校验']:
                    # '已校验'/'未校验' 对应 status 字段且 confidence != -1
                    db_status_val = STATUS_MAP_FRONTEND_TO_DB.get(status_cn)
                    if db_status_val:
                        current_status_q_objects.append(Q(status=db_status_val) & ~Q(confidence=-1))

            if current_status_q_objects:
                # 使用 reduce(operator.or_, ...) 来组合多个状态条件 (Q1 OR Q2 OR Q3)
                final_q_conditions.append(reduce(operator.or_, current_status_q_objects))

        # 应用所有筛选条件
        if final_q_conditions:
            queryset = queryset.filter(reduce(operator.and_, final_q_conditions))

        # 只有 confidence 不为 -1 的样本才会被更新 status
        # 这是为了确保 "重置校验" 和 "完成校验" 按钮不会改变人工标注样本的状态
        queryset_to_update = queryset.exclude(confidence=-1)

        updated_count = queryset_to_update.update(status=status_to_set_en)

        return JsonResponse({
            'status': 'success',
            'message': f'批量更新成功，共更新了 {updated_count} 条记录。',
            'updated_count': updated_count
        })

    except Exception as e:
        logger.error(f"批量更新任务 {task_id} 时出错: {str(e)}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'服务器内部错误: {str(e)}'}, status=500)


def export_data(request, task_id):
    """统一数据导出"""
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user)

    config = TASK_CONFIG[task.task_type]
    model = config['result_model']

    buffer = StringIO()
    buffer.write(codecs.BOM_UTF8.decode('utf-8'))
    writer = csv.writer(buffer)

    # 写入表头
    writer.writerow([col[1] for col in config['csv_fields']])

    # 写入数据
    for item in model.objects.filter(**{config['relation_field']: task}):
        # 获取字段值并进行格式化，例如置信度
        row_values = []
        for field_name, _ in config['csv_fields']:
            value = getattr(item, field_name)
            if field_name == 'confidence' and value is not None:
                # 导出时也格式化置信度，-1 特殊处理
                value = f"{float(value):.3f}" if float(value) != -1.0 else "-1.000"
            elif field_name == 'status':
                # 将内部状态转换为前端显示的状态
                is_manual_annotation = (item.confidence == -1 or item.confidence == -1.0)
                if is_manual_annotation:
                    value = '已标注'
                else:
                    value = STATUS_MAP_DB_TO_FRONTEND.get(value, value) # 转换为中文
            row_values.append(value)
        writer.writerow(row_values)

    response = HttpResponse(
        buffer.getvalue().encode('utf-8-sig'),
        content_type='text/csv; charset=utf-8-sig'
    )
    filename = f"{task_id}_{task.task_type}_results.csv"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


@csrf_exempt
@require_http_methods(["POST"])
def verify_item(request, task_id):
    """
    统一验证操作：根据前端传入的 target_status 设置状态。
    如果 target_status 是 '已校验'，则设置为已校验；
    如果 target_status 是 '未校验'，则设置为未校验 (相当于取消校验)。
    """
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user)

    config = TASK_CONFIG[task.task_type]
    model = config['result_model']
    item_id = request.POST.get('item_id')
    target_status_cn = request.POST.get('target_status') # 前端传入的中文目标状态

    if not item_id or not target_status_cn:
        return JsonResponse(
            {'status': 'error', 'message': '缺少必要参数 (item_id 或 target_status)'},
            status=400
        )

    try:
        item = model.objects.get(id=item_id)

        # 映射前端中文状态到后端英文状态
        target_status_en = STATUS_MAP_FRONTEND_TO_DB.get(target_status_cn)
        # 注意：这里从前端传过来就应该是 '已校验' 或 '未校验'，不应该出现 '已标注'
        if not target_status_en or target_status_en not in ['verified', 'unverified']:
            return JsonResponse(
                {'status': 'error', 'message': '无效的目标状态'},
                status=400
            )

        # 如果当前状态已经是目标状态，则无需修改
        if item.status == target_status_en:
            return JsonResponse(
                {'status': 'no_change', 'message': f'状态已是 {target_status_cn}'},
                status=200
            )

        item.status = target_status_en
        item.save()

        message = f'状态已更新为 {target_status_cn}'
        return JsonResponse({'status': 'success', 'message': message})

    except model.DoesNotExist:
        return JsonResponse(
            {'status': 'error', 'message': '记录不存在'},
            status=404
        )
    except Exception as e:
        logger.error(f"验证项失败 (task_id: {task_id}, item_id: {item_id}): {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'服务器内部错误: {str(e)}'}, status=500)


@login_required
def task_data(request, task_id):
    """
    获取任务相关的分页数据，并支持筛选、排序功能。
    """
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user)

    config = TASK_CONFIG[task.task_type]
    result_model = config['result_model']
    relation_field = config['relation_field']

    page = int(request.GET.get('page', 1))
    page_size = int(request.GET.get('page_size', 10)) # 确保与前端 pageSize 变量一致

    # 从URL获取并解码参数
    labels_raw = urllib.parse.unquote(request.GET.get('label', ''))
    confidences_raw = urllib.parse.unquote(request.GET.get('confidence', ''))
    statuses_raw = urllib.parse.unquote(request.GET.get('status', '')) # 前端传递的中文状态列表

    labels = [l.strip() for l in labels_raw.split(',') if l.strip()]
    confidences = [c.strip() for c in confidences_raw.split(',') if c.strip()]
    statuses_cn = [s.strip() for s in statuses_raw.split(',') if s.strip()]
    labeled_only = request.GET.get('labeled_only') == '1'
    predicted_only = request.GET.get('predicted_only') == '1'

    sort = request.GET.get('sort', '')

    queryset = result_model.objects.filter(**{relation_field: task})
    uncertainty_scores = {}
    if task.task_type == 'image-classification':
        uncertainty_scores = _load_image_uncertainty_scores(task)
    elif task.task_type == 'text-classification':
        uncertainty_scores = _load_text_uncertainty_scores(task)

    if task.task_type == 'text-classification' and not task.label_file and not predicted_only:
        return JsonResponse({
            'results': [],
            'page': 1,
            'total_pages': 1,
            'total_count': 0,
        })

    if task.task_type == 'text-classification' and not task.label_file and not predicted_only:
        return JsonResponse({
            'results': [],
            'page': 1,
            'total_pages': 1,
            'total_count': 0,
        })

    # 构建一个列表来存储所有独立的 Q 对象，最后用 AND 组合它们
    final_q_conditions = []

    # 1. 标签筛选 (多选 OR 逻辑)
    if labels:
        final_q_conditions.append(Q(label__in=labels))

    # 2. 置信度筛选 (多选 OR 逻辑)
    if confidences:
        numeric_confidences = []
        for c_str in confidences:
            try:
                # 将前端的 "-1.000" 映射回数据库的 -1.0
                if c_str == '-1.000':
                    numeric_confidences.append(-1.0)
                else:
                    numeric_confidences.append(float(c_str))
            except ValueError:
                logger.warning(f"task_data: 跳过无效置信度筛选值: {c_str}")
                pass # 跳过无效的置信度字符串
        if numeric_confidences:
            # 使用 Q(confidence__in=numeric_confidences) 可以更简洁地处理 OR 逻辑
            final_q_conditions.append(Q(confidence__in=numeric_confidences))

    # 2.5 仅已标注 / 仅模型预测筛选
    if labeled_only:
        final_q_conditions.append(Q(confidence=-1))
    if predicted_only:
        final_q_conditions.append(~Q(confidence=-1))

    # 3. 状态筛选 (多选 OR 逻辑 - 关键修改部分)
    if statuses_cn:
        current_status_q_objects = []
        for status_cn in statuses_cn:
            if status_cn == '已标注':
                # '已标注' 对应 confidence = -1
                current_status_q_objects.append(Q(confidence=-1))
            elif status_cn in ['已校验', '未校验']:
                # '已校验'/'未校验' 直接对应 status 字段
                db_status_val = STATUS_MAP_FRONTEND_TO_DB.get(status_cn)
                if db_status_val: # 确保映射成功
                    current_status_q_objects.append(Q(status=db_status_val))
            # 忽略前端传递的未知状态，避免错误

        if current_status_q_objects:
            # 使用 reduce(operator.or_, ...) 来组合多个状态条件 (Q1 OR Q2 OR Q3)
            final_q_conditions.append(reduce(operator.or_, current_status_q_objects))

    # 应用所有筛选条件 (AND 逻辑)
    if final_q_conditions:
        queryset = queryset.filter(reduce(operator.and_, final_q_conditions))

    if task.task_type == 'text-classification' and not task.label_file:
        queryset = queryset.order_by('sequence_number', 'created_at', 'id')


    # 应用排序条件
    if sort:
        column, order = sort.split(':')
        if column in ('uncertainty', 'entropy') and task.task_type in ('image-classification', 'text-classification'):
            queryset_list = list(queryset)

            def _uncertainty_sort_key(item):
                score_key = getattr(item, 'image_name', '') if task.task_type == 'image-classification' else getattr(item, 'text_id', '')
                base_score = uncertainty_scores.get(score_key, None)
                if base_score is None:
                    try:
                        base_score = max(0.0, 1.0 - float(item.confidence))
                    except (TypeError, ValueError):
                        base_score = 0.0
                return base_score

            queryset_list.sort(key=_uncertainty_sort_key, reverse=(order == 'desc'))
            queryset = queryset_list
        elif column == 'confidence':
            # 将 confidence 字段转换为浮点数进行排序，并处理 -1.0 的特殊情况（通常排在最前或最后）
            # Django的orderBy默认会按数字大小排序，-1.0会排在前面，符合预期
            order_by_clause = f'-{column}' if order == 'desc' else column
            queryset = queryset.order_by(order_by_clause, '-created_at', '-id')
        elif column == 'label':
            if order == 'desc':
                queryset = queryset.order_by(Lower(F(column)).desc(), '-created_at', '-id')
            else:
                queryset = queryset.order_by(Lower(F(column)), '-created_at', '-id')
        elif column == 'status':
            # 对状态进行自定义排序：已校验 > 未校验 > 已标注 (confidence=-1)
            # 使用 Case/When 来定义一个临时的排序值
            # 优先处理 confidence = -1 的 '已标注' 状态
            queryset = queryset.annotate(
                status_order=Case(
                    When(confidence=-1.0, then=Value(3)), # 已标注 (confidence=-1) 排在最后 (值越大越靠后)
                    When(status='verified', then=Value(1)), # 已校验 (非人工) 排在最前
                    When(status='unverified', then=Value(2)), # 未校验 (非人工) 排在中间
                    default=Value(4), # 默认值，以防有其他未知状态，排在最后
                    output_field=FloatField() # 确保是数值类型
                )
            )
            # 如果是降序，则数值大的在前；升序，数值小的在前
            order_by_clause = '-status_order' if order == 'desc' else 'status_order'
            queryset = queryset.order_by(order_by_clause, '-created_at', '-id')
        else:
            # 未知排序字段回退到“最新优先”
            queryset = queryset.order_by('-created_at', '-id')
    else:
        # 默认按进入标注界面的顺序（最新进入在最前）
        queryset = queryset.order_by('-created_at', '-id')

    paginator = Paginator(queryset, page_size)

    try:
        page_obj = paginator.page(page)
    except EmptyPage:
        # 如果请求的页码超出范围，则返回空结果但总页数和总数应正确
        return JsonResponse({
            'results': [],
            'page': paginator.num_pages, # 返回实际的最后一页或第一页
            'total_pages': paginator.num_pages,
            'total_count': paginator.count
        })

    results = []
    for item in page_obj.object_list:
        is_manual_annotation = (item.confidence == -1 or item.confidence == -1.0) # 明确判断是否是人工标注

        # 状态统一只返回“已校验/未校验”，是否人工标注通过额外字段表示。
        display_status = STATUS_MAP_DB_TO_FRONTEND.get(item.status, item.status)

        # 格式化置信度，人工标注的显示为 "-1.000"
        display_confidence = f"{float(item.confidence):.3f}" if item.confidence is not None and item.confidence != -1 else "-1.000"
        if task.task_type == 'text-classification' and not task.label_file:
            display_confidence = ''

        result = {
            'id': str(item.id), # 将 UUID 转换为字符串
            'label': item.label,
            'confidence': display_confidence,
            'status': display_status,
            'sequence_number': item.sequence_number,
            'created_at': item.created_at.strftime('%Y-%m-%d %H:%M:%S') if item.created_at else None
        }
        if isinstance(item, TextResult):
            result['content'] = item.content
            # 只有在非人工标注的样本中，才在无标签任务中清空标签
            if task.task_type == 'text-classification' and not task.label_file and not is_manual_annotation:
                result['label'] = ''
                result['confidence'] = ''
            text_uncertainty = uncertainty_scores.get(item.text_id)
            if text_uncertainty is None:
                try:
                    # 不确定性：仅当非人工标注时才计算；人工标注的不确定性为 0
                    text_uncertainty = max(0.0, 1.0 - float(item.confidence)) if item.confidence != -1 else 0.0
                except (TypeError, ValueError):
                    text_uncertainty = 0.0
            result['uncertainty_score'] = round(float(text_uncertainty), 6)
        elif isinstance(item, ImageResult):
            result['relative_image_url'] = item.relative_image_url
            result['image_name'] = item.image_name
            result['is_manual_annotation'] = is_manual_annotation
            uncertainty_score = uncertainty_scores.get(item.image_name)
            if uncertainty_score is None:
                try:
                    uncertainty_score = max(0.0, 1.0 - float(item.confidence)) if item.confidence != -1 else 0.0
                except (TypeError, ValueError):
                    uncertainty_score = 0.0
            result['uncertainty_score'] = round(float(uncertainty_score), 6)
        results.append(result)

    data = {
        'results': results,
        'page': page_obj.number,
        'total_pages': paginator.num_pages,
        'total_count': paginator.count
    }

    return JsonResponse(data)


# 获取单个任务的最新标注统计数据
@login_required
@require_http_methods(["GET"])
def get_task_stats(request, task_id):
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user)

    config = TASK_CONFIG[task.task_type]
    result_model = config['result_model']
    relation_field = config['relation_field']

    stats = _get_label_verification_stats(task, result_model, relation_field)
    total_count = result_model.objects.filter(**{relation_field: task}).count()

    # 如果数据库中尚未写入样本（例如任务刚提交，后台尚未处理），
    # 则回退到根据上传的解压目录估算总样本数，便于前端立刻显示上传数量。
    if total_count == 0 and getattr(task, 'extracted_dir', None):
        try:
            extract_path = Path(task.extracted_dir)
            if extract_path.exists():
                # 图片任务：按常见图片扩展名计数
                if task.task_type == 'image-classification':
                    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
                    total_count = sum(1 for p in extract_path.rglob('*') if p.suffix.lower() in exts)
                elif task.task_type == 'text-classification':
                    # 文本任务：统计所有 csv 文件的行数（除首行表头）
                    total_count = 0
                    for p in extract_path.rglob('*.csv'):
                        try:
                            with open(p, 'r', encoding='utf-8', newline='') as f:
                                reader = csv.reader(f)
                                rows = list(reader)
                                if len(rows) > 1:
                                    total_count += max(0, len(rows) - 1)
                        except Exception:
                            continue
                else:
                    # 兜底：计数所有文件（排除隐藏）
                    total_count = sum(1 for p in extract_path.rglob('*') if p.is_file() and not p.name.startswith('.'))
        except Exception:
            # 出错时保留原始 total_count（0）
            total_count = result_model.objects.filter(**{relation_field: task}).count()

    total_verifiable = stats.get('total_verifiable_count', total_count)
    verified = stats.get('verified_count', 0)
    unverified = stats.get('unverified_count', 0)
    manual_annotated_count = stats.get('manual_annotated_count', stats.get('total_labeled_count', 0))
    verification_progress = stats.get('verification_progress', 0)

    # 返回JSON格式的统计数据
    return JsonResponse({
        'status': 'success',
        'stats': {
            'verified_count': verified,
            'unverified_count': unverified,
            'manual_annotated_count': manual_annotated_count,
            'total_count': total_count,
            'total_verifiable_count': total_verifiable,
            'verification_progress': verification_progress
        }
    })


@login_required
@require_POST
def toggle_favorite(request, task_id):
    """切换收藏状态"""
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user) # 确保用户有权限收藏该任务

    # 检查是否已收藏
    favorite = FavoriteTask.objects.filter(
        user=request.user,
        task=task
    ).first()

    if favorite:
        # 已收藏则取消
        favorite.delete()
        is_favorite = False
        message = '已取消收藏'
    else:
        # 未收藏则添加，并确保不超过10个
        favorites_count = FavoriteTask.objects.filter(user=request.user).count()
        if favorites_count >= 10:
            # 删除最早的收藏
            oldest = FavoriteTask.objects.filter(user=request.user).order_by('created_at').first()
            if oldest:
                oldest.delete()

        # 添加新收藏
        FavoriteTask.objects.create(user=request.user, task=task)
        is_favorite = True
        message = '已收藏任务'

    return JsonResponse({
        'is_favorite': is_favorite,
        'isFavorite': is_favorite,  # 兼容管理页面
        'message': message,
        'task_id': task_id
    })


@login_required
def check_favorite(request, task_id):
    """检查收藏状态"""
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user) # 确保用户有权限检查该任务的收藏状态

    is_favorite = FavoriteTask.objects.filter(
        user=request.user,
        task=task
    ).exists()
    return JsonResponse({'is_favorite': is_favorite})


@login_required
def get_favorites(request):
    """获取收藏列表"""
    try:
        favorites = FavoriteTask.objects.filter(
            user=request.user
        ).select_related('task').order_by('-created_at')[:10]

        favorite_list = [{
            'id': str(fav.task.id), # UUID 转换为字符串
            'name': fav.task.name,
            'type': fav.task.get_task_type_display(),
            'favorited_at': fav.created_at.strftime('%Y-%m-%d %H:%M'),
            'url': f"/tasks/{fav.task.id}/detail/" # 为收藏项添加一个可跳转的 URL
        } for fav in favorites]

        return JsonResponse({
            'status': 'success',
            'favorites': favorite_list
        })

    except Exception as e:
        logger.error(f"获取收藏列表失败: {e}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e),
            'favorites': []  # 即使出错也返回空数组
        }, status=500)

@login_required
def get_favorite_status(request, task_id):
    """获取收藏状态"""
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user) # 确保用户有权限获取该任务的收藏状态

    is_favorite = FavoriteTask.objects.filter(
        user=request.user,
        task=task
    ).exists()
    return JsonResponse({'is_favorite': is_favorite})


@csrf_exempt
@require_POST
def confirm_denoise(request, task_id):
    """
    接收前端去噪确认请求，批量更新对应项的标签和状态。
    只会更新非人工标注（confidence != -1）的样本。
    """
    try:
        data = json.loads(request.body)
        logger.info(f"Denoise confirmation request for task {task_id}: {len(data)} items")

        if not isinstance(data, list):
            return JsonResponse({'status': 'error', 'message': '请求体格式错误，需要一个JSON数组'}, status=400)

        task_obj = get_object_or_404(Task, id=task_id)
        TaskService.validate_ownership(task_obj, request.user)

        config = TASK_CONFIG[task_obj.task_type]
        result_model = config['result_model']

        success_count = 0
        total_attempted = 0

        with transaction.atomic():
            for item_data in data:
                # 检查 item_data 结构，防止无效请求
                if not isinstance(item_data, dict) or 'id' not in item_data or 'label' not in item_data or 'status' not in item_data:
                    logger.warning(f"跳过无效的去噪项数据: {item_data}")
                    continue

                total_attempted += 1
                try:
                    obj = result_model.objects.get(id=item_data['id'])

                    # **核心逻辑：确保去噪结果的确认只作用于非人工标注的样本**
                    if obj.confidence == -1:
                        logger.warning(f"Item {item_data['id']} 是人工标注样本 (confidence=-1)，跳过其去噪结果确认。")
                        continue

                    # 将前端传入的中文状态映射为后端数据库期望的英文状态
                    target_status_en = STATUS_MAP_FRONTEND_TO_DB.get(item_data['status'])
                    if not target_status_en or target_status_en not in ['verified', 'unverified']:
                        logger.warning(f"Item {item_data['id']} 目标状态无效或未映射: {item_data['status']}")
                        continue # 跳过无效状态

                    # 如果标签和状态都未改变，则不进行实际的数据库更新
                    if obj.label == item_data['label'] and obj.status == target_status_en:
                        logger.debug(f"Item {item_data['id']} 标签和状态未变化，跳过更新。")
                        success_count += 1 # 视为成功处理，因为无需改变
                        continue

                    obj.label = item_data['label']
                    obj.status = target_status_en
                    obj.save()
                    success_count += 1
                    logger.debug(f"成功更新 Item {item_data['id']} 至标签: {obj.label}, 状态: {obj.status}")

                except result_model.DoesNotExist:
                    logger.warning(f"Item ID = {item_data['id']} 不存在，跳过。")
                except Exception as e:
                    logger.error(f"处理去噪项 {item_data.get('id', 'N/A')} 时发生错误: {e}", exc_info=True)
                    # 继续处理下一个，不中断事务
                    # 这里也可以选择回滚整个事务并返回错误，取决于业务需求
                    # 如果希望即使有一个失败也继续，则不raise，并在catch中处理日志
                    pass # 这里的 pass 表示捕获异常但不重新抛出，继续循环

        return JsonResponse({
            'status': 'success',
            'message': f'成功应用了 {success_count} / {total_attempted} 条去噪结果。'
        })
    except json.JSONDecodeError:
        logger.error("Denoise confirmation: Invalid JSON in request body.")
        return JsonResponse({'status': 'error', 'message': '无效的JSON格式'}, status=400)
    except Task.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '任务不存在'}, status=404)
    except PermissionDenied as e:
        logger.warning(f"Denoise confirmation: Permission denied for user {request.user.id} on task {task_id}.")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=403)
    except Exception as e:
        logger.exception("Unexpected error during denoise confirmation.") # 使用 exception 记录完整堆栈
        return JsonResponse({'status': 'error', 'message': f'服务器内部错误: {str(e)}'}, status=500)


# 新增一个智能去噪的模拟API，实际应该调用您的模型去噪逻辑
@login_required
@require_POST
def perform_denoise(request, task_id): 
    """
    触发一个异步的智能去噪任务，让前端轮询任务状态并等待去噪结果。
    """
    # task_id 现在是整数，可以直接使用
    task = get_object_or_404(Task, id=task_id)
    TaskService.validate_ownership(task, request.user)

    config = TASK_CONFIG[task.task_type]
    result_model = config['result_model']
    relation_field = config['relation_field']

    try:
        # 简单模拟去噪逻辑：查找非人工标注且置信度较低的项作为“噪声”
        # 实际应替换为您的AI模型去噪逻辑
        potential_noise_items = result_model.objects.filter(
            **{relation_field: task}
        ).exclude(confidence=-1).filter(confidence__lt=0.7).order_by('confidence')[:20] # 假设取20个低置信度项

        denoised_suggestions = []
        for item in potential_noise_items:
            # 模拟一个“去噪建议”标签，例如在原标签前加“修正_”
            # 真实情况可能需要更复杂的逻辑或模型的建议标签
            suggested_label = f"修正_{item.label}" if item.label else "修正_未知标签"
            denoised_suggestions.append({
                'id': str(item.id), # 注意：这里 ID 依然需要是字符串，因为可能是 UUID Field
                'new_label': suggested_label
            })

        return JsonResponse({
            'status': 'success',
            'message': '智能去噪分析完成。',
            'denoised_suggestions': denoised_suggestions
        })
    except Exception as e:
        logger.error(f"执行智能去噪时出错: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'服务器内部错误: {str(e)}'}, status=500)
