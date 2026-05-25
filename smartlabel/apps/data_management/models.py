from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.utils.html import format_html
from django.db.models import JSONField
import uuid

class Task(models.Model):
    STATUS_CHOICES = (
        ('pending', '等待中'),
        ('processing', '进行中'),
        ('completed', '已完成'),
        ('checking', '校验中'),
        ('failed', '创建失败'),
    )
    TYPE_CHOICES = (
        ('image-classification', '图像分类'),
        ('text-classification', '文本分类'),
        ('object-detection', '目标检测'),
    )
    LABELING_CHOICES = (
        ('pre-trained', '预训练'),
        ('semi-supervised', '半监督'),
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    # 基础配置信息（用户提交信息）
    name = models.CharField(max_length=100, verbose_name="任务名称")
    description = models.TextField(blank=True, verbose_name="任务描述")
    task_type = models.CharField(max_length=20, choices=TYPE_CHOICES, verbose_name="任务类型")
    labeling_type = models.CharField(max_length=20, choices=LABELING_CHOICES, verbose_name="标注模式",default="pre-trained")
    # 文本分类下的场景选择：情感分析 / 主题分类 / 内容审核
    CLASSIFICATION_SCENE_CHOICES = (
        ('sentiment', '情感分析'),
        ('topic', '主题分类'),
        ('moderation', '内容审核'),
    )
    classification_scene = models.CharField(max_length=30, choices=CLASSIFICATION_SCENE_CHOICES, blank=True, null=True, verbose_name="分类场景")

    # 数据和模型配置信息
    data_file = models.FileField(upload_to='uploads/data/%Y/%m/%d/', verbose_name="数据文件")
    # 用户上传数据时统计的固定总数（上传完成后写入，界面显示为常量）
    uploaded_total = models.PositiveIntegerField(default=0, verbose_name="上传总数")
    label_list = models.JSONField(default=list, blank=True, verbose_name="标签配置")   # 针对预训练模式
    label_file = models.FileField(upload_to='uploads/label/%Y/%m/%d/', blank=True, null=True,
                                  verbose_name="标注文件")  # 半监督模式下必填
    extracted_dir = models.CharField(max_length=255, blank=True, null=True, verbose_name="解压文件")
    model_choice = models.CharField(max_length=50, blank=True, null=True, verbose_name="模型版本")
    model_strength = models.CharField(max_length=50, blank=True, null=True, verbose_name="模型强度")

    # 创建与传递信息
    celery_task_id = models.CharField(max_length=50, blank=True, null=True, verbose_name="celery任务ID")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    # 处理进程信息
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name="任务状态")
    progress = models.PositiveIntegerField(default=0, verbose_name="处理进度")
    error_log = models.TextField(blank=True, null=True, verbose_name="错误日志")

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['status', 'task_type', 'labeling_type']),
            models.Index(fields=['name']),
            models.Index(fields=['created_at']),
            models.Index(fields=['updated_at']),
        ]

    def __str__(self):
        return f"{self.name}({self.get_task_type_display()})"


class ImageResult(models.Model):
    # 定义 ImageResult 自己的状态 Choices
    ITEM_STATUS_CHOICES = (
        ('verified', '已校验'),
        ('unverified', '未校验'),
        # 注意：'已标注' 状态是通过 confidence=-1 来区分的，不是 status 字段的直接值
        # 这里只包含 status 字段实际存储的两种状态
    )

    sequence_number = models.PositiveIntegerField(default=1, verbose_name="任务内序号")
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='results')
    image_name = models.CharField(max_length=512, default="img")
    image_path = models.CharField(max_length=512)
    label = models.CharField(max_length=255)
    confidence = models.FloatField()
    annotations = JSONField(default=list, blank=True, null=True, verbose_name='目标检测标注')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=ITEM_STATUS_CHOICES, default="unverified")

    def save(self, *args, **kwargs):
        # 如果是新创建的实例，且 sequence_number 尚未设置（默认值为1）
        if not self.pk and self.sequence_number == 1:
            # 获取当前任务下最大的 sequence_number
            max_seq = ImageResult.objects.filter(task=self.task).aggregate(models.Max('sequence_number'))['sequence_number__max']
            if max_seq is not None:
                self.sequence_number = max_seq + 1
            else:
                self.sequence_number = 1 # 如果没有记录，则从1开始
        super().save(*args, **kwargs)

    def image_data(self):
        return format_html(
            '<img src="{}" width="100px"/>',
            self.image_path.url,
        )

    image_data.short_description = u'图片'

    def __str__(self):
        # 可以在这里显示任务内部序号
        return f"{self.task.name} - {self.sequence_number} - {self.image_name}"

    @property
    def relative_image_url(self):
        if not self.image_path:
            return ''  # 如果数据库里是空的，避免报错
        path = self.image_path.replace("\\", "/")  # 统一用斜杠
        path = path.replace(settings.MEDIA_ROOT.replace("\\", "/"), "")  # 去掉绝对路径
        path = path.lstrip("/")  # 去掉开头多余的 /
        return f"/media/{path}"

    class Meta:
        indexes = [
            models.Index(fields=['task', 'label']),
            models.Index(fields=['task', 'sequence_number']), # 增加索引以优化查询
        ]
        # 确保每个任务内的 sequence_number 是唯一的
        unique_together = ('task', 'sequence_number',)


class TextResult(models.Model):
    # 定义 TextResult 自己的状态 Choices
    ITEM_STATUS_CHOICES = (
        ('verified', '已校验'),
        ('unverified', '未校验'),
        # 同上，这里不包含 '已标注'
    )

    sequence_number = models.PositiveIntegerField(default=1, verbose_name="任务内序号")
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='text_results')
    text_id = models.CharField(max_length=512, verbose_name="文本ID")
    content = models.TextField(verbose_name="文本内容")
    label = models.CharField(max_length=255)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=ITEM_STATUS_CHOICES, default="unverified")

    def save(self, *args, **kwargs):
        # 如果是新创建的实例，且 sequence_number 尚未设置（默认值为1）
        if not self.pk and self.sequence_number == 1:
            # 获取当前任务下最大的 sequence_number
            max_seq = TextResult.objects.filter(task=self.task).aggregate(models.Max('sequence_number'))['sequence_number__max']
            if max_seq is not None:
                self.sequence_number = max_seq + 1
            else:
                self.sequence_number = 1 # 如果没有记录，则从1开始
        super().save(*args, **kwargs)

    class Meta:
        indexes = [
            models.Index(fields=['task', 'label']),
            models.Index(fields=['task', 'sequence_number']), # 增加索引以优化查询
        ]
        # 确保每个任务内的 sequence_number 是唯一的
        unique_together = ('task', 'sequence_number',)

    def __str__(self):
        # 可以在这里显示任务内部序号
        return f"{self.task.name} - {self.sequence_number} - {self.text_id} - {self.content[:30]}"


class FavoriteTask(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='favorite_tasks')
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='favorited_by')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'task')  # 确保每个用户对每个任务只能收藏一次
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username}收藏的任务: {self.task.name}"

