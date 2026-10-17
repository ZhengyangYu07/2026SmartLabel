"""项目包入口：导出 Celery 应用，保证 Django 启动时一并加载 Celery。"""
from .celery import app as celery_app

__all__ = ('celery_app',)