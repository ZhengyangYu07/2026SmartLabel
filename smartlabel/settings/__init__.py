# 兜底配置入口：直接继承 base（base 未定义 SECRET_KEY，仅供 asgi/wsgi 默认引用）。
# 实际运行由 manage.py 与 celery.py 指定 smartlabel.settings.dev。
from .base import *