"""
数据管理应用的 Django Admin 配置。

把核心模型注册到后台并定制列表/筛选/搜索，便于运维排查：
Task、ImageResult、TextResult，以及半监督闭环的三张表
（PositiveIndex 正样本索引、MissFixTable 漏检补框、ReviewQueueCache 复核队列缓存）。
"""
from django.contrib import admin

from .models import MissFixTable, PositiveIndex, ReviewQueueCache, Task, ImageResult, TextResult


@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    # 列表显示的字段
    list_display = (
        'name', 'task_type', 'status', 'user', 'model_choice', 'model_strength', 'created_at', 'updated_at',
        'labeling_type',
        'progress', 'celery_task_id'
    )

    # 可以在列表页直接编辑的字段
    list_editable = ('status', 'progress', 'celery_task_id')

    # 列表页的筛选器
    list_filter = ('status', 'task_type', 'created_at')

    # 搜索字段
    search_fields = ('name', 'description', 'user__username')

    # 详细页面的字段分组
    fieldsets = (
        ('基本信息', {
            'fields': ('name', 'description', 'task_type', 'labeling_type', 'user')
        }),
        ('文件信息', {
            'fields': ('data_file', 'label_file', 'extracted_dir')
        }),
        ('模型信息', {
            'fields': ('model_choice', 'model_strength')
        }),
        ('处理进程信息', {
            'fields': ('status', 'progress', 'created_at', 'updated_at', 'celery_task_id', 'error_log')
        }),
    )

    # 只读字段（在详细页面中不可编辑）
    readonly_fields = ('created_at', 'updated_at', 'model_choice', 'model_strength')

    # 按创建时间降序排列
    ordering = ('-created_at',)


@admin.register(ImageResult)
class ImageResultAdmin(admin.ModelAdmin):
    list_display = ('image_name', 'label', 'confidence', 'status', 'image_path')
    list_filter = ('label', 'status')  # 指定可以过滤的字段
    search_fields = ('image_name', 'label')  # 指定可以搜索的字段


@admin.register(TextResult)
class TextResultAdmin(admin.ModelAdmin):
    list_display = ('label', 'confidence', 'status', 'content')
    list_filter = ('label', 'status')  # 指定可以过滤的字段
    search_fields = ('content', 'label')  # 指定可以搜索的字段


@admin.register(PositiveIndex)
class PositiveIndexAdmin(admin.ModelAdmin):
    list_display = ('task', 'image_name', 'class_label', 'source_type', 'index_version', 'is_active', 'created_at')
    list_filter = ('source_type', 'is_active')
    search_fields = ('image_name', 'class_label', 'sample_id')


@admin.register(MissFixTable)
class MissFixTableAdmin(admin.ModelAdmin):
    list_display = ('task', 'sample_id', 'manual_class', 'index_write_status', 'positive_index_version', 'fixed_at')
    list_filter = ('index_write_status',)
    search_fields = ('sample_id', 'manual_class')


@admin.register(ReviewQueueCache)
class ReviewQueueCacheAdmin(admin.ModelAdmin):
    list_display = ('task', 'sample_id', 'branch_type', 'review_score', 'cache_status', 'queue_version', 'index_version_snapshot')
    list_filter = ('branch_type', 'cache_status')
    search_fields = ('sample_id', 'trigger_reason')
