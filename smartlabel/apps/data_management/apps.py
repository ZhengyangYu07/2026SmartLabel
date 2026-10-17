"""数据管理应用的 AppConfig（应用名 data_management）。"""
from django.apps import AppConfig


class DataManagementConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'data_management'
