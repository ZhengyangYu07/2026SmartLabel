import os
import sys

# 步骤1: 设置环境变量，必须在最前面
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartlabel.settings.dev')

from celery import Celery

# 步骤2: 创建Celery app实例
app = Celery('smartlabel')

# =========================================================================
# 关键修复：将通用配置加载移出 if 判断块
# -------------------------------------------------------------------------
# 这两行代码对于“生产者”(如Django视图)和“消费者”(Celery Worker)都至关重要。
# 必须保证任何导入 app 的地方都能加载到 settings.py 中的配置。
# 这样，当视图调用 task.delay() 时，才知道要将任务发送到 Redis。
# =========================================================================
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现所有INSTALLED_APPS下定义的tasks.py文件
app.autodiscover_tasks()


# =========================================================================
# Worker进程专属配置
# -------------------------------------------------------------------------
# 以下代码块仅在启动Celery Worker时执行。
# 它包含了猴子补丁、信号处理等所有对 runserver 进程有害或无用的操作。
# =========================================================================
is_celery_worker = 'celery' in sys.argv or 'worker' in sys.argv
is_eventlet_pool = (
    'eventlet' in sys.argv
    or any(arg.startswith('--pool=eventlet') for arg in sys.argv)
)

if is_celery_worker:
    # --- 猴子补丁：仅当显式使用 eventlet 池时才应用 ---
    if is_eventlet_pool:
        import eventlet
        eventlet.monkey_patch()
        print("Eventlet monkey patch applied for Celery worker.")

    # --- 仅在Worker进程中加载的模块 ---
    import logging
    import time
    import psutil
    import torch
    import django
    from celery.signals import worker_process_init, task_failure, task_success
    from django.conf import settings as django_settings

    # --- Django 初始化 (对于Worker是必须的) ---
    django.setup()
    print("Django has been initialized for the Celery worker.")

    # 获取Celery专用的日志记录器
    celery_logger = logging.getLogger('celery')

    # ---- 信号处理函数 ----
    @worker_process_init.connect
    def configure_worker_process(**kwargs):
        import gc
        gc.set_threshold(700, 10, 10)
        celery_logger.info("Worker process initialized successfully.")
        # 在worker子进程启动时，清理一次可能残留的旧进程
        cleanup_old_training_processes()

    @task_failure.connect
    def task_failure_handler(sender=None, task_id=None, exception=None, einfo=None, **kwargs):
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
                task.error_message = error_msg
                task.save(update_fields=['status', 'error_message'])
        except Exception as e:
            celery_logger.error(f"Error updating failed task status: {e}")

    @task_success.connect
    def task_success_handler(sender=None, result=None, **kwargs):
        if hasattr(sender, 'request') and hasattr(sender.request, 'id'):
            task_id = sender.request.id
            celery_logger.info(f"Task completed successfully: {task_id}")
            cleanup_gpu_memory()

    # ---- 辅助函数 ----
    def cleanup_old_training_processes():
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
                    
                    # 清理条件：运行时间超过软限制 或 成为僵尸进程
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
                            pass # Process already gone
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
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                celery_logger.info("GPU memory cache cleared.")
        except ImportError:
            pass # torch not available
        except Exception as e:
            celery_logger.error(f"Error while cleaning up GPU memory: {e}")

    def cleanup_task_related_processes(task_id):
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
        try:
            # Check Django settings
            if not django_settings.configured:
                print("ERROR: Django settings are not configured!")
                return False
            
            # Check Database connection
            from django.db import connection
            connection.ensure_connection()
            print("✓ Database connection is OK.")
            
            # Check Memory usage
            memory_info = monitor_memory_usage()
            if memory_info:
                print(f"✓ Current memory usage: {memory_info['percent']:.1f}%")
            
            print("✓ Celery startup health check passed.")
            return True
        except Exception as e:
            print(f"✗ Celery startup health check failed: {e}")
            return False

    # --- 调试模式配置 ---
    try:
        if django_settings.DEBUG:
            app.conf.update(
                task_eager_propagates_exceptions=True,
            )
            print("DEBUG mode is active. Tasks will execute immediately and raise exceptions.")
    except Exception:
        pass

    # --- 启动时输出关键配置信息 ---
    print("Celery worker starting up with key settings:")
    print(f"- Broker URL: {app.conf.get('broker_url')}")
    print(f"- Result Backend: {app.conf.get('result_backend')}")
    print(f"- Soft time limit: {app.conf.get('task_soft_time_limit', 'Not Set')} seconds")
    print(f"- Hard time limit: {app.conf.get('task_time_limit', 'Not Set')} seconds")
    print(f"- Worker max memory: {app.conf.get('worker_max_memory_per_child', 0) / 1024:.1f} MB")
    print(f"- Max tasks per child: {app.conf.get('worker_max_tasks_per_child', 'Unlimited')}")
    print(f"- Prefetch multiplier: {app.conf.get('worker_prefetch_multiplier', 'Default')}")
    
    # --- 执行启动健康检查 ---
    startup_health_check()
