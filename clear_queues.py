import argparse

from redis import Redis

from smartlabel.celery import app


def purge_all_queues():
    """清空当前 worker 监听的所有队列。"""
    purged = app.control.purge()
    print(f"已清空队列中的消息数: {purged}")


def flush_result_backend():
    """清空 Celery 结果后端(仅支持 Redis)。"""
    backend_url = app.conf.result_backend
    if not backend_url or not str(backend_url).startswith("redis://"):
        print(f"结果后端不是 Redis，跳过清理: {backend_url}")
        return

    client = Redis.from_url(backend_url)
    key_count = client.dbsize()
    client.flushdb()
    print(f"已清空结果后端 Redis，删除键数量: {key_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理 Celery 队列和结果后端缓存")
    parser.add_argument(
        "--skip-results",
        action="store_true",
        help="仅清空队列，不清空结果后端 Redis",
    )
    args = parser.parse_args()

    purge_all_queues()
    if not args.skip_results:
        flush_result_backend()
