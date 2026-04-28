from django.urls import path

from . import views

urlpatterns = [
    path('', views.welcome, name='welcome'),
    path('user_login/', views.user_login, name='user_login'),
    path('user_register/', views.user_register, name='user_register'),
    path('user_forget', views.user_change, name='user_forget'),
    path('homepage/', views.homepage, name='homepage'),
    # 用户手册
    path('user_guide/introduction/', views.user_guide_introduction, name='user_guide'),
    path('user_guide/basicfunction/login_logout/', views.user_guide_basicfunction_login_logout, name='user_guide_basicfunction_login_logout'),
    path('user_guide/basicfunction/change_information/', views.user_guide_basicfunction_change_information, name='user_guide_basicfunction_change_information'),
    path('user_guide/homepage/overview/', views.user_guide_homepage_overview, name='user_guide_homepage_overview'),
    path('user_guide/homepage/log/', views.user_guide_homepage_log, name='user_guide_homepage_log'),
    path('user_guide/homepage/return/', views.user_guide_homepage_return, name='user_guide_homepage_return'),
    path('user_guide/task/ceate/', views.user_guide_task_create, name='user_guide_task_create'),
    path('user_guide/task/ceate/', views.user_guide_task_manage, name='user_guide_task_manage'),
    path('user_guide/confirm/image/classify', views.user_guide_confirm_image_classify, name='user_guide_confirm_image_classify'),
    path('user_guide/confirm/text/classify', views.user_guide_confirm_text_classify, name='user_guide_confirm_text_classify'),
    path('user_guide/faq/', views.user_guide_faq, name='user_guide_faq'),
    # 个人
    path('profile/', views.profile, name='profile'),
    path('change_password/', views.change_password, name='change_password'),
    path('create_task/', views.create_task, name='create_task'),
    path('manage_task/', views.manage_task, name='manage_task'),
    path('feedback/', views.feedback, name='feedback'),
    path('feedback/submit/', views.submit_feedback, name='submit_feedback'),


]
