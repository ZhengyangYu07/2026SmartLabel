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

# 任务显示内容配置
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
}


class TaskService:
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
