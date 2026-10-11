"""
Celery 应用配置。

职责：
1. 创建 Celery 实例，并从 Django settings 中以 CELERY_ 前缀加载全部配置；
2. 自动发现各 app 下的 tasks.py；
3. 仅在 Worker 进程中执行初始化（猴子补丁、信号回调、进程与显存清理）。

本模块被 smartlabel/__init__.py 导入，因此 Django 启动时会自动加载。
"""
import os
import sys

# 必须在导入 Celery/Django 之前设置，保证任何进程都能定位到配置文件
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartlabel.settings.dev')

from celery import Celery

# 创建 Celery 应用实例
app = Celery('smartlabel')

# 从 django.conf:settings 加载 CELERY_ 开头的配置。
# 生产者（Django 视图调用 task.delay）与消费者（Worker）都依赖这一步，因此不能放进下面的 if。
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现各 INSTALLED_APPS 下的 tasks.py
app.autodiscover_tasks()


# 以下仅在当前进程是 Celery Worker 时执行；
# 猴子补丁、信号、进程清理等对 runserver 进程无意义甚至有害。
is_celery_worker = 'celery' in sys.argv or 'worker' in sys.argv
is_eventlet_pool = (
    'eventlet' in sys.argv
    or any(arg.startswith('--pool=eventlet') for arg in sys.argv)
)

if is_celery_worker:
    # 仅在显式使用 eventlet 池时打猴子补丁（Windows 推荐 threads 池，见 README）
    if is_eventlet_pool:
        import eventlet
        eventlet.monkey_patch()
        print("Eventlet monkey patch applied for Celery worker.")

    # 以下模块只在 Worker 进程内加载
    import logging
    import time
    import psutil
    import torch
    import django
    from celery.signals import worker_process_init, task_failure, task_success
    from django.conf import settings as django_settings

    # Worker 进程内同样需要初始化 Django，才能使用 ORM
    django.setup()
    print("Django has been initialized for the Celery worker.")

    celery_logger = logging.getLogger('celery')

    # ------------------------------------------------------------------
    # 信号回调
    # ------------------------------------------------------------------
    @worker_process_init.connect
    def configure_worker_process(**kwargs):
        """Worker 子进程启动时：调整 GC 阈值，并清理上次残留的训练进程。"""
        import gc
        gc.set_threshold(700, 10, 10)
        celery_logger.info("Worker process initialized successfully.")
        cleanup_old_training_processes()

    @task_failure.connect
    def task_failure_handler(sender=None, task_id=None, exception=None, einfo=None, **kwargs):
        """任务异常时：释放显存、清理相关子进程，并把对应 Task 标记为 failed。"""
        celery_logger.error(f"Task failed {task_id}: {exception}")
        cleanup_gpu_memory()
        if task_id:
            cleanup_task_related_processes(task_id)

        try:
            from data_management.models import Task
            task = Task.objects.filter(id=task_id).first()
            if task:
                task.status = 'failed'
                error_msg = str(exception)[:500] if exception else "Unknown error"
                if "out of memory" in error_msg.lower():
                    error_msg = "GPU内存不足，建议使用CPU模式或减少批次大小"
                # 注意：Task 模型的实际字段名是 error_log，此处 error_message 与之不一致，
                # save() 会抛 ValueError 并被下面的 except 捕获（不影响主流程）。
                task.error_message = error_msg
                task.save(update_fields=['status', 'error_message'])
        except Exception as e:
            celery_logger.error(f"Error updating failed task status: {e}")

    @task_success.connect
    def task_success_handler(sender=None, result=None, **kwargs):
        """任务成功后释放显存。"""
        if hasattr(sender, 'request') and hasattr(sender.request, 'id'):
            task_id = sender.request.id
            celery_logger.info(f"Task completed successfully: {task_id}")
            cleanup_gpu_memory()

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------
    def cleanup_old_training_processes():
        """清理超时或已僵尸化的算法训练子进程。

        只处理命令行含 text_classification/flexmatch/img_classfication 的 python 进程；
        运行时长超过 CELERY_TASK_SOFT_TIME_LIMIT 或处于僵尸状态时终止。
        """
        try:
            current_pid = os.getpid()
            cleaned_count = 0
            current_time = time.time()
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'status']):
                try:
                    if proc.info['pid'] == current_pid or proc.info['name'] != 'python':
                        continue
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    is_training_process = any([
                        'text_classification/train.py' in cmdline,
                        'flexmatch/train.py' in cmdline,
                        'img_classfication/train.py' in cmdline,
                    ])
                    if not is_training_process:
                        continue

                    process_age = current_time - proc.info['create_time']
                    process_age_hours = process_age / 3600
                    should_cleanup = False
                    cleanup_reason = ""

                    # 清理条件：运行时间超过软限制 或 已成为僵尸进程
                    if process_age_hours > django_settings.CELERY_TASK_SOFT_TIME_LIMIT / 3600:
                        should_cleanup = True
                        cleanup_reason = f"Running too long ({process_age_hours:.1f} hours)"
                    elif proc.info['status'] == psutil.STATUS_ZOMBIE:
                        should_cleanup = True
                        cleanup_reason = "Zombie process"

                    if should_cleanup:
                        celery_logger.info(f"Cleaning up old training process: PID={proc.info['pid']}, Reason={cleanup_reason}, Cmd={cmdline[:100]}...")
                        try:
                            p = psutil.Process(proc.info['pid'])
                            p.terminate()
                            p.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            p.kill()
                            celery_logger.warning(f"Forcefully killed process {proc.info['pid']}")
                        except psutil.NoSuchProcess:
                            pass  # 进程已自行退出
                        cleaned_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                except Exception as e:
                    celery_logger.error(f"Error while inspecting process: {e}")
                    continue

            if cleaned_count > 0:
                celery_logger.info(f"Cleaned up {cleaned_count} old training processes.")
                cleanup_gpu_memory()
            else:
                celery_logger.info("No old training processes found to clean up.")
        except Exception as e:
            celery_logger.error(f"Error during cleanup of old training processes: {e}")

    def cleanup_gpu_memory():
        """清空 PyTorch 的 CUDA 缓存；无 GPU 时静默跳过。"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                celery_logger.info("GPU memory cache cleared.")
        except ImportError:
            pass  # torch 不可用
        except Exception as e:
            celery_logger.error(f"Error while cleaning up GPU memory: {e}")

    def cleanup_task_related_processes(task_id):
        """终止指定任务残留的训练子进程（命令行含 results/<task_id>/）。"""
        try:
            cleaned_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline_str = ' '.join(proc.info['cmdline'] or [])
                    if (f'results/{task_id}/' in cmdline_str and
                        'python' in (proc.info['name'] or '') and
                        'train.py' in cmdline_str):

                        celery_logger.info(f"Cleaning up subprocess for task {task_id}: PID={proc.info['pid']}")
                        try:
                            p = psutil.Process(proc.info['pid'])
                            p.terminate()
                            p.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass
                        cleaned_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if cleaned_count > 0:
                celery_logger.info(f"Cleaned up {cleaned_count} subprocess(es) for task {task_id}.")
        except Exception as e:
            celery_logger.error(f"Error cleaning up subprocesses for task {task_id}: {e}")

    def monitor_memory_usage():
        """返回 Worker 进程内存占用；占用率超过 80% 时告警。"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            if memory_percent > 80:
                celery_logger.warning(f"High worker memory usage detected: {memory_percent:.1f}%")
            return {'rss': memory_info.rss, 'vms': memory_info.vms, 'percent': memory_percent}
        except ImportError:
            celery_logger.warning("psutil is not installed. Cannot monitor memory usage.")
            return None
        except Exception as e:
            celery_logger.error(f"Error during memory usage monitoring: {e}")
            return None

    def startup_health_check():
        """Worker 启动自检：确认 settings 已配置、数据库可连接，并打印内存占用。"""
        try:
            if not django_settings.configured:
                print("ERROR: Django settings are not configured!")
                return False

            from django.db import connection
            connection.ensure_connection()
            print("✓ Database connection is OK.")

            memory_info = monitor_memory_usage()
            if memory_info:
                print(f"✓ Current memory usage: {memory_info['percent']:.1f}%")

            print("✓ Celery startup health check passed.")
            return True
        except Exception as e:
            print(f"✗ Celery startup health check failed: {e}")
            return False

    # 调试模式下让任务同步执行并抛出异常，便于定位问题
    try:
        if django_settings.DEBUG:
            app.conf.update(
                task_eager_propagates_exceptions=True,
            )
            print("DEBUG mode is active. Tasks will execute immediately and raise exceptions.")
    except Exception:
        pass

    # 启动时打印关键配置，便于排查
    print("Celery worker starting up with key settings:")
    print(f"- Broker URL: {app.conf.get('broker_url')}")
    print(f"- Result Backend: {app.conf.get('result_backend')}")
    print(f"- Soft time limit: {app.conf.get('task_soft_time_limit', 'Not Set')} seconds")
    print(f"- Hard time limit: {app.conf.get('task_time_limit', 'Not Set')} seconds")
    print(f"- Worker max memory: {app.conf.get('worker_max_memory_per_child', 0) / 1024:.1f} MB")
    print(f"- Max tasks per child: {app.conf.get('worker_max_tasks_per_child', 'Unlimited')}")
    print(f"- Prefetch multiplier: {app.conf.get('worker_prefetch_multiplier', 'Default')}")

    # 执行启动健康检查
    startup_health_check()