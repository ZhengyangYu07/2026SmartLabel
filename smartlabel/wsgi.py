"""
WSGI 部署入口，模块级变量 application 即 WSGI 可调用对象。

注意：这里指向 smartlabel.settings（即 base，未定义 SECRET_KEY），
如需以 WSGI 部署，应改指向 settings.dev 或补全 settings.prov。
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartlabel.settings')

application = get_wsgi_application()