document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('create-form');
    if (!form) {
        console.error("创建任务的表单 '#create-form' 未找到！");
        return;
    }

    const saveDraftBtn = document.getElementById('save-draft');
    if (saveDraftBtn) {
        saveDraftBtn.addEventListener('click', () => {
            const formData = new FormData(form);
            // 暂存草稿时，删除文件数据，因为文件不能直接暂存
            formData.delete('data_file');
            formData.delete('label_file');

            // 尝试获取 Alpine.js 实例，以便使用其消息显示功能
            const alpineData = form.__x ? form.__x.data : null;

            fetch('/api/save_draft/', {
                method: 'POST',
                body: formData,
                headers: { 'X-CSRFToken': form.elements.csrfmiddlewaretoken.value }
            })
            .then(res => res.json())
            .then(data => {
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
