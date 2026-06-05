"""
Django base settings for smartlabel project.

包含所有环境的通用配置，不应包含敏感信息
"""

import os
import sys
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 添加应用目录到Python路径
sys.path.insert(0, str(BASE_DIR / 'smartlabel/apps'))
sys.path.insert(0, str(BASE_DIR))

# 应用定义
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',

    'smartlabel.apps.users.apps.UsersConfig',
    'smartlabel.apps.data_management.apps.DataManagementConfig'
]

# 中间件
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# 模板配置
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media'
            ],
        },
    },
]

# 数据库配置
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# 密码验证
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# 国际化
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_TZ = True

# 静态文件
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'collected_static')
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

# 媒体文件
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

# 默认字段类型
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# 认证相关
LOGIN_URL = '/user_login/'
LOGIN_REDIRECT_URL = '/homepage/'

# 其他全局配置
ROOT_URLCONF = 'smartlabel.urls'
WSGI_APPLICATION = 'smartlabel.wsgi.application'

# =========================================================================
# 关键修改: 集中化的Celery配置
# 所有配置都放在这里，并使用 CELERY_ 前缀。
# =========================================================================
CELERY_BROKER_URL = 'redis://127.0.0.1:6379/0'
CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/1'
CELERY_TIMEZONE = 'Asia/Shanghai'

# === 序列化配置 ===
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'

# === 任务执行与持久化 ===
CELERY_TASK_ACKS_LATE = True
CELERY_TASK_REJECT_ON_WORKER_LOST = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

# === 超时设置 ===
CELERY_TASK_SOFT_TIME_LIMIT = 7200  # 2小时软限制
CELERY_TASK_TIME_LIMIT = 10800      # 3小时硬限制

# === 内存管理 ===
CELERY_WORKER_MAX_MEMORY_PER_CHILD = 2000000  # 2GB
CELERY_WORKER_MAX_TASKS_PER_CHILD = 20

# === 连接与重试 ===
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
CELERY_BROKER_CONNECTION_RETRY = True
CELERY_BROKER_CONNECTION_MAX_RETRIES = 10
CELERY_BROKER_POOL_LIMIT = 10

CELERY_BROKER_TRANSPORT_OPTIONS = {
    'visibility_timeout': 7200,
    'socket_connect_timeout': 30,
    'socket_keepalive': True,
    'socket_timeout': 30,
}

CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS = {
    'retry_policy': {
        'timeout': 10.0,
        'max_retries': 3,
    }
}

# === 任务发布重试 ===
CELERY_TASK_PUBLISH_RETRY = True
CELERY_TASK_PUBLISH_RETRY_POLICY = {
    'max_retries': 3,
    'interval_start': 0.5,
    'interval_step': 0.5,
    'interval_max': 3,
}

# === 结果后端 ===
CELERY_RESULT_EXPIRES = 7200  # 结果2小时后过期

# === 任务路由配置 ===
from kombu import Queue, Exchange

# 1. 定义所有你会用到的队列
CELERY_TASK_QUEUES = (
    Queue('default', Exchange('default'), routing_key='default'),
    Queue('annotation_tasks', Exchange('annotation_tasks'), routing_key='annotation_tasks'),
)

# 2. 明确指定默认队列，以防万一
CELERY_TASK_DEFAULT_QUEUE = 'default'
CELERY_TASK_DEFAULT_EXCHANGE = 'default'
CELERY_TASK_DEFAULT_ROUTING_KEY = 'default'

# 3. 定义任务到队列的路由规则
#    这是核心，它告诉Celery哪个任务应该去哪个队列
CELERY_TASK_ROUTES = {
    'data_management.tasks.process_annotation_task': {
        'queue': 'annotation_tasks',
        'routing_key': 'annotation_tasks',
    },
    # 所有其他未明确指定的任务，都将进入 'default' 队列
    # 'data_management.tasks.*': {'queue': 'default'}, # 这行可以保留也可以注释，因为上面已经设置了默认队列
}

# 4. 确保在Broker中不存在队列时，Celery会自动创建它
CELERY_TASK_CREATE_MISSING_QUEUES = True

# === 监控和调试 ===
CELERY_WORKER_SEND_TASK_EVENTS = True
CELERY_TASK_SEND_SENT_EVENT = True
CELERY_TASK_IGNORE_RESULT = False

# === 日志配置 ===
CELERY_WORKER_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
CELERY_WORKER_TASK_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s[%(task_name)s][%(task_id)s]: %(message)s'
CELERY_WORKER_LOG_COLOR = False
CELERY_WORKER_HIJACK_ROOT_LOGGER = False

# === 其他性能配置 ===
CELERY_WORKER_DISABLE_RATE_LIMITS = True
CELERY_WORKER_HEARTBEAT = 30


# =========================================================================
# 关键修改: 日志配置
# =========================================================================
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'smartlabel.log'),
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'data_management': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False, # 推荐设置为False，避免其日志重复
        },
        'celery': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            # 关键修复：设置为False，防止日志向root logger传播，解决重复打印问题。
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}
