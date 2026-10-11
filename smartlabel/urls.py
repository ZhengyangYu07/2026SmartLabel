"""
项目根路由。

分发关系：
- admin/            Django 后台
- '' → users.urls   登录/注册/首页/用户手册等
- '' → data_management.urls  任务创建、进度、详情、导出等接口

末尾 static() 用于开发环境直接提供 /media/ 访问，DEBUG=False 时自动失效（返回空列表）。
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
                  path('admin/', admin.site.urls),
                  path('', include('users.urls')),
                  path('', include('data_management.urls')),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)