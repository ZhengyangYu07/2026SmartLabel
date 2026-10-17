"""
ASGI 部署入口，模块级变量 application 即 ASGI 可调用对象。

注意：这里指向 smartlabel.settings（即 base，未定义 SECRET_KEY），
如需以 ASGI 部署，应改指向 settings.dev 或补全 settings.prov。
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartlabel.settings')

application = get_asgi_application()