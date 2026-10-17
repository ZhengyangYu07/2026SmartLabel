"""Django 命令行管理入口，用法：python manage.py <子命令>（如 runserver / migrate）。"""
import os
import sys


def main():
    """设置默认配置模块，并把命令行参数交给 Django 执行。"""
    # 默认使用开发环境配置；生产部署应改为 smartlabel.settings.prov
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartlabel.settings.dev')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()