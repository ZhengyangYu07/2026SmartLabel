/**
 * 创建任务页（create_task）脚本。
 * 负责“暂存草稿”按钮的交互：把表单内容（剔除文件字段）通过 AJAX 提交到 /api/save_draft/，
 * 成功或失败时优先用 Alpine.js 的消息组件提示，不可用时回退到 alert。
 */
document.addEventListener('DOMContentLoaded', () => {
    // 页面初始化：定位创建任务表单，缺失则直接退出
    const form = document.getElementById('create-form');
    if (!form) {
        console.error("创建任务的表单 '#create-form' 未找到！");
        return;
    }

    // 绑定“暂存草稿”按钮的点击事件
    const saveDraftBtn = document.getElementById('save-draft');
    if (saveDraftBtn) {
        saveDraftBtn.addEventListener('click', () => {
            // 收集表单数据
            const formData = new FormData(form);
            // 暂存草稿时，删除文件数据，因为文件不能直接暂存
            formData.delete('data_file');
            formData.delete('label_file');

            // 尝试获取 Alpine.js 实例，以便使用其消息显示功能
            const alpineData = form.__x ? form.__x.data : null;

            // AJAX 提交草稿到后端接口（携带 CSRF token）
            fetch('/api/save_draft/', {
                method: 'POST',
                body: formData,
                headers: { 'X-CSRFToken': form.elements.csrfmiddlewaretoken.value }
            })
            .then(res => res.json())
            .then(data => {
                // 按后端返回状态给出成功或失败反馈
                if(data.status === 'success') {
                    if (alpineData && typeof alpineData.displayValidationMessage === 'function') {
                        alpineData.displayValidationMessage('草稿已暂存！', 'success');
                    } else {
                        alert('草稿已暂存！'); // Alpine数据不可用时回退
                    }
                } else {
                    if (alpineData && typeof alpineData.displayValidationMessage === 'function') {
                        alpineData.displayValidationMessage('暂存失败: ' + data.message, 'error');
                    } else {
                        alert('暂存失败: ' + data.message); // Alpine数据不可用时回退
                    }
                }
            })
            .catch(err => {
                if (alpineData && typeof alpineData.displayValidationMessage === 'function') {
                    alpineData.displayValidationMessage('暂存请求失败: ' + err.message, 'error');
                } else {
                    alert('暂存请求失败: ' + err.message); // Alpine数据不可用时回退
                }
            });
        });
    }
});
