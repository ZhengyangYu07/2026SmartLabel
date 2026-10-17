"""
开发环境配置：继承 base 并覆盖仅供本地调试使用的选项。
"""
from .base import *

# 密钥硬编码仅限开发使用，生产环境必须改为从环境变量读取
SECRET_KEY = 'django-insecure-w8e68cd=%-gz$02))aqnsu!c&$klnx)d*49k8r)!88m@eh3+qt'

# 调试开关：开启后报错页含堆栈、静态/媒体文件由 Django 直接提供
DEBUG = True

# 开发环境放开所有 Host，生产环境需收紧为具体域名
ALLOWED_HOSTS = ['*']

# 开发环境数据库如需覆盖，在此重定义 DATABASES

# 邮件改为输出到控制台，便于本地调试
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'