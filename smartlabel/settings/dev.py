"""
Development specific settings

继承自 base.py 并覆盖开发环境配置
"""
from .base import *

# 安全警告：开发环境专用配置
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-w8e68cd=%-gz$02))aqnsu!c&$klnx)d*49k8r)!88m@eh3+qt'  # 生产环境应从环境变量获取

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']  # 开发环境允许所有主机

# 开发环境数据库配置（如果与base不同可覆盖）
# DATABASES = {...}

# 开发环境特殊中间件（如DebugToolbar）
# MIDDLEWARE += ['debug_toolbar.middleware.DebugToolbarMiddleware']
# INSTALLED_APPS += ['debug_toolbar']

# 邮件配置（开发环境使用控制台输出）
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
