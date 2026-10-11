# 生产/线上环境配置（占位文件）。
# 当前仅继承 base；正式部署时需在此覆盖：SECRET_KEY（从环境变量读取）、DEBUG=False、
# ALLOWED_HOSTS、生产数据库，以及 Redis/Celery 的真实地址。
from .base import *