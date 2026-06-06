from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash


def welcome(request):
    return render(request, 'enter/welcome.html')


def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()

        # 基础参数校验
        if not username or not password:
            return render(request, 'enter/user_login.html', {
                'error_message': "用户名和密码不能为空",
                'username': username
            })

        # 身份验证
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # 跳转到next参数或默认首页
            next_url = request.POST.get('next') or 'homepage'
            return redirect(next_url)
        else:
            # 返回错误信息给模板
            return render(request, 'enter/user_login.html', {
                'error_message': "用户名或密码错误",
                'username': username
            })

    # GET请求携带next参数处理
    context = {'next': request.GET.get('next', '')}
    return render(request, 'enter/user_login.html', context)


def user_register(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '').strip()
        confirm_password = request.POST.get('confirm_password', '').strip()

        # 分步校验提高可读性
        error_messages = []

        # 必填字段校验
        if not all([username, email, password, confirm_password]):
            error_messages.append("所有字段均为必填项")

        # 密码一致性校验
        elif password != confirm_password:
            error_messages.append("两次输入的密码不一致")

        # 密码复杂度校验（可根据需求扩展）
        elif len(password) < 8:
            error_messages.append("密码长度不能小于8位")

        # 用户名唯一性校验
        if User.objects.filter(username__iexact=username).exists():
            error_messages.append("用户名已被占用")

        # 邮箱格式及唯一性校验
        if '@' not in email:
            error_messages.append("邮箱格式不正确")
        elif User.objects.filter(email__iexact=email).exists():
            error_messages.append("该邮箱已被注册")

        if error_messages:
            return render(request, 'enter/user_register.html',
                          {'error_messages': list(set(error_messages))})

        try:
            # 创建用户并自动登录
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )
            login(request, user)
            messages.success(request, "注册成功！")
            return redirect('homepage')

        except Exception as e:
            # 捕获未知异常
            error_messages.append("系统错误，请稍后再试")
            return render(request, 'enter/user_register.html',
                          {'error_messages': error_messages})

    return render(request, 'enter/user_register.html')


def homepage(request):
    return render(request, 'platform/homepage.html')


#######################################################################################################################
def user_guide_introduction(request):
    return render(request, 'documentation/user_guide_introduction.html')


def user_guide_basicfunction_login_logout(request):
    return render(request, 'documentation/user_guide_basicfunction_login_logout.html')


def user_guide_basicfunction_change_information(request):
    return render(request, 'documentation/user_guide_basicfunction_change_information.html')


def user_guide_homepage_overview(request):
    return render(request, 'documentation/user_guide_homepage_overview.html')


def user_guide_homepage_log(request):
    return render(request, 'documentation/user_guide_homepage_log.html')


def user_guide_homepage_return(request):
    return render(request, 'documentation/user_guide_homepage_return.html')


def user_guide_task_create(request):
    return render(request, 'documentation/user_guide_task_create.html')


def user_guide_task_manage(request):
    return render(request, 'documentation/user_guide_task_manage.html')


def user_guide_confirm_image_classify(request):
    return render(request, 'documentation/user_guide_confirm_image_classify.html')


def user_guide_confirm_text_classify(request):
    return render(request, 'documentation/user_guide_confirm_text_classify.html')


def user_guide_faq(request):
    return render(request, 'documentation/user_guide_faq.html')


#######################################################################################################################


def profile(request):
    return render(request, "platform/accounts/profile.html")

@login_required
def change_password(request):
    if request.method == 'POST':
        old_password = request.POST.get('old_password')
        new_password1 = request.POST.get('new_password1')
        new_password2 = request.POST.get('new_password2')
        user = request.user

        # 检查旧密码是否正确
        if not user.check_password(old_password):
            return render(request, 'platform/accounts/change_password.html', {
                'error_message': '旧密码错误，请重新输入。'
            })

        # 检查新密码两次是否一致
        if new_password1 != new_password2:
            return render(request, 'platform/accounts/change_password.html', {
                'error_message': '两次输入的新密码不一致，请重新输入。'
            })

        # 设置新密码
        user.set_password(new_password1)
        user.save()
        update_session_auth_hash(request, user)  # 重要，防止修改密码后登出=

        return render(request, 'platform/accounts/change_password.html', {
            'success_message': '密码修改成功！'
        })
    return render(request, "platform/accounts/change_password.html")

def user_change(request):
    return render(request, "enter/user_change.html")


def create_task(request):
    return render(request, "platform/tasks/create_task.html")


def manage_task(request):
    return render(request, "platform/tasks/manage_task.html")


def feedback(request):
    return render(request, "platform/accounts/feedback.html")


def submit_feedback(request):
    if request.method == 'POST':
        # 你可以将反馈保存到数据库，或发送邮件等
        messages.success(request, '感谢您的反馈！')
        return redirect('feedback')  # 返回原反馈页面
    return render(request, "platform/accounts/feedback.html")  # 渲染原页面
