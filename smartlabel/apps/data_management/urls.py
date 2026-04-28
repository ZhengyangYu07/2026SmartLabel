from django.urls import path

from . import views

urlpatterns = [
    # 创建任务路由配置
    path('api/get_draft/', views.get_draft, name='get_draft'),
    path('api/save_draft/', views.save_draft, name='save_draft'),
    path('api/submit_task/', views.submit_task, name='submit_task'),
    path('api/clear_draft/', views.clear_draft, name='clear_draft'),

    # 获取首页统计数据路由配置
    path('api/homepage_stats/', views.get_homepage_stats, name='api_homepage_stats'),

    # 数据文件校验路由配置
    path('api/validate_data_file/', views.validate_data_file, name='validate_data_file'),

    # 获取任务列表路由配置
    path('api/tasks/', views.get_tasks, name='get_tasks'),
    path('api/tasks/<int:task_id>/', views.task_operation, name='task_operation'),
    path('api/tasks/<int:task_id>/progress/', views.task_progress, name='task_progress'),
    path('tasks/<int:task_id>/unlabeled/', views.unlabeled_detail, name='unlabeled_detail'),
    path('tasks/<int:task_id>/manual_annotation/', views.manual_annotation, name='manual_annotation'),
    path('tasks/<int:task_id>/manual_annotation/save/', views.save_manual_annotation, name='save_manual_annotation'),
    path('tasks/<int:task_id>/manual_annotation/add_unlabeled/', views.add_unlabeled_to_labeled, name='add_unlabeled_to_labeled'),
    path('tasks/<int:task_id>/manual_annotation/rerun/', views.rerun_semi_supervised, name='rerun_semi_supervised'),

    path('api/tasks/<int:task_id>/favorite/', views.toggle_favorite, name='toggle_favorite'),
    path('api/tasks/<int:task_id>/check_favorite/', views.check_favorite, name='check_favorite'),
    path('api/favorites/', views.get_favorites, name='get_favorites'),
    path('api/tasks/<int:task_id>/favorite/status/', views.get_favorite_status, name='get_favorite_status'),

    # 查看任务详情路由配置
    path('api/tasks/<int:task_id>/stats/', views.get_task_stats, name='get_task_stats'),
    path('tasks/<int:task_id>/detail/', views.task_detail, name='task_detail'),
    path('api/tasks/<int:task_id>/meta_data/', views.get_task_meta_data, name='get_task_meta_data'),
    path('tasks/<int:task_id>/update/', views.handle_label_update, name='update_label'),
    path('tasks/<int:task_id>/updateall/', views.update_task_items, name='update_task_items'),
    path('tasks/<int:task_id>/export/', views.export_data, name='export_data'),
    path('tasks/<int:task_id>/verify/', views.verify_item, name='verify_item'),
    path('tasks/<int:task_id>/data/', views.task_data, name='task_data'),
    path('api/tasks/<int:task_id>/denoise/', views.perform_denoise, name='perform_denoise'),
    path('tasks/<int:task_id>/confirm_denoise/', views.confirm_denoise, name='confirm_denoise'),
]
