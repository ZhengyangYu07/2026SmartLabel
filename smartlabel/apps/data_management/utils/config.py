"""
任务详情/导出相关的公共配置与工具。

TASK_CONFIG 用“任务类型 → 结果模型、详情模板、CSV 字段”的方式，把三类任务
（图像分类、文本分类、目标检测）的差异集中到一处，视图层据此通用化处理。
"""
import codecs
import csv
import json
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from io import StringIO
from django.core.exceptions import PermissionDenied

from ..models import Task, ImageResult, TextResult

# 任务显示内容配置：新增任务类型时在此登记即可，详情页与导出逻辑无需改动
TASK_CONFIG = {
    'image-classification': {
        'result_model': ImageResult,
        'detail_template': 'platform/tasks/image_detail_h.html',
        'csv_fields': [
            ('id', 'ID'),
            ('image_name', '文件名'),
            ('image_path', '文件路径'),
            ('label', '预测标签'),
            ('confidence', '置信度'),
            ('status', '校验状态')
        ],
        'relation_field': 'task'
    },
    'text-classification': {
        'result_model': TextResult,
        'detail_template': 'platform/tasks/text_detail_h.html',
        'csv_fields': [
            ('id', 'ID'),
            ('content', '内容'),
            ('label', '预测标签'),
            ('confidence', '置信度'),
            ('status', '校验状态')
        ],
        'relation_field': 'task'
    }
    ,
    'object-detection': {
        'result_model': ImageResult,
        'detail_template': 'platform/tasks/detection_detail_h.html',
        'csv_fields': [
            ('id', 'ID'),
            ('image_name', '文件名'),
            ('image_path', '文件路径'),
            ('annotations', '标注(JSON)'),
            ('status', '校验状态')
        ],
        'relation_field': 'task'
    }
}


def display_task_progress(status, progress):
    """任务未真正完成时，前端最多展示到 99%。"""
    try:
        progress_value = float(progress or 0)
    except (TypeError, ValueError):
        progress_value = 0
    progress_value = max(0, min(progress_value, 100))
    if status != 'completed' and progress_value >= 100:
        return 99
    return progress_value


class TaskService:
    """任务详情页的公共逻辑：拼装上下文、校验归属。"""

    def __init__(self):
        pass

    @staticmethod
    def get_common_context(task):
        """获取公共上下文数据"""
        config = TASK_CONFIG[task.task_type]
        result_model = config['result_model']
        relation_field = config['relation_field']
        total_count = result_model.objects.filter(**{relation_field: task}).count()
        task.total_count = total_count
        # 进度经 display_task_progress 处理，未完成时不会显示 100%
        task.progress = display_task_progress(task.status, task.progress)
        return {
            'task': task,
            'total_tasks': Task.objects.count(),
            'total_verified': Task.objects.filter(status='verified').count(),
            'config': config,
            'task_total_count': total_count,
        }

    @staticmethod
    def validate_ownership(task, user):
        """验证任务所有权"""
        if task.user != user:
            raise PermissionDenied("无权访问此任务")
