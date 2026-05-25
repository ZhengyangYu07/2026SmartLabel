document.addEventListener("DOMContentLoaded", () => {
  const TaskManager = {
    config: {
      tasksPerPage: 12,
      currentPage: 1,
      totalPages: 0,
      pollingIntervals: new Map(),
    },

    statusColors: {
      pending: "bg-gray-400",
      processing: "bg-blue-400",
      checking: "bg-yellow-400",
      completed: "bg-green-400",
      failed: "bg-red-400",
    },

    statusTexts: {
      pending: "等待中",
      processing: "进行中",
      checking: "校验中",
      completed: "已完成",
      failed: "失败",
    },

    taskTypes: {
      "image-classification": "图像分类",
      "text-classification": "文本分类",
      "object-detection": "目标检测",
    },

    filters: {
      selectedStatus: [],
      taskType: "",
      taskName: "",
    },

    // 收藏功能
    toggleFavorite: async function (taskId, starElement) {
      console.log("toggleFavorite 被调用，参数:", { taskId, starElement });
      try {
        const response = await fetch(`/api/tasks/${taskId}/favorite/`, {
          method: "POST",
          headers: {
            "X-CSRFToken": this.getCSRFToken(),
            "Content-Type": "application/json",
          },
        });

        console.log("响应状态:", response.status);
        console.log("响应头:", response.headers);

        if (!response.ok) {
          throw new Error(`请求失败: ${response.status}`);
        }

        const data = await response.json();
        console.log("服务器返回数据:", data);

        const isFavorite = data.is_favorite;

        // 更新星标显示
        this.updateStarDisplay(starElement, isFavorite);

        // 显示操作反馈
        this.showToast(
          data.message || (isFavorite ? "已收藏任务" : "已取消收藏"),
          isFavorite ? "success" : "info",
        );

        // 核心逻辑：如果任务被收藏，将其移动到列表最前面
        if (isFavorite) {
          const taskList = document.getElementById("project-list");
          const taskCard = starElement.closest(".task-card");
          if (taskCard && taskList && taskList.firstChild !== taskCard) {
            // 动画效果：先缩小，再移动，再放大
            taskCard.style.transition =
              "transform 0.3s ease-in-out, opacity 0.3s ease-in-out";
            taskCard.style.transform = "scale(0.9)";
            taskCard.style.opacity = "0";

            // 延迟移除和插入，以便看到动画
            setTimeout(() => {
              // 确保卡片仍是父级的子元素，避免重复添加或错误移除
              if (taskCard.parentNode === taskList) {
                taskList.removeChild(taskCard);
                taskList.prepend(taskCard); // 将卡片移到最前面
              }
              // 恢复动画属性和状态
              taskCard.style.transition =
                "transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out";
              taskCard.style.transform = "scale(1)";
              taskCard.style.opacity = "1";
            }, 300); // 300ms 动画完成后执行移动
          }
        } else {
          // 如果是取消收藏，则重新加载列表以恢复默认排序（因为取消收藏后它就不再是最高优先级了）
          // 也可以选择不刷新，但这样会打破默认的收藏优先排序规则
          // 考虑到页面可能有多页，刷新列表是更稳妥的方式来恢复后端排序逻辑
          this.updateTaskList();
        }

        // 发送全局消息通知其他页面
        this.broadcastFavoriteChange(taskId, isFavorite);

        return data;
      } catch (error) {
        console.error("切换收藏状态失败:", error);
        this.showToast("操作失败，请重试", "error");
        throw error;
      }
    },
    // 新增：统一星标显示更新
    updateStarDisplay: function (starElement, isFavorite) {
      if (isFavorite) {
        starElement.classList.add("is-favorite"); // 添加新的CSS类
        starElement.style.color = "";
        starElement.innerHTML = "★"; // 实心星
        starElement.title = "取消收藏";
      } else {
        starElement.classList.remove("is-favorite"); // 移除新的CSS类
        starElement.style.color = ""; // 灰色
        starElement.innerHTML = "☆"; // 空心星
        starElement.title = "收藏此任务";
      }
    },

    // 广播收藏状态变化
    broadcastFavoriteChange: function (taskId, isFavorite) {
      // 通知其他页面更新收藏列表
      if (typeof updateSavedTasksList === "function") {
        updateSavedTasksList();
      }

      // 发送消息给其他页面同步状态
      window.postMessage(
        {
          type: "favoriteChanged",
          taskId: taskId,
          isFavorite: isFavorite,
        },
        "*",
      );
    },
    // 显示提示消息
    showToast: function (message, type = "info") {
      const toast = document.createElement("div");
      // 完全使用base_task.html中的样式
      toast.className = `fixed top-20 left-1/2 -translate-x-1/2 px-5 py-3 rounded-md shadow-lg transition-opacity duration-300 ease-in-out z-5000 text-white bg-black border border-gray-600`;

      // 添加图标前缀
      if (type === "success") {
        toast.innerHTML = "✓ " + message;
      } else if (type === "error") {
        toast.innerHTML = "✗ " + message;
      } else {
        toast.innerHTML = "⚠ " + message;
      }

      document.body.appendChild(toast);

      // 添加淡出动画
      setTimeout(() => {
        toast.style.opacity = "0";
        toast.addEventListener(
          "transitionend",
          () => {
            toast.remove();
          },
          { once: true },
        );
      }, 1500);
    },

    // 筛选任务方法
    bindFilterEvents() {
      // 初始化时显示所有任务
      this.filters.selectedStatus = [];
      this.updateButtonStatus();
      this.updateFilterStatus();

      // 状态按钮点击事件优化
      document.querySelectorAll(".status-btn").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          const btnElement = e.target.closest(".status-btn");
          const status = btnElement.dataset.status;
          const isActive = btnElement.classList.contains("active");

          // 当点击按钮后，切换当前按钮状态
          if (isActive) {
            // 移除状态
            this.filters.selectedStatus = this.filters.selectedStatus.filter(
              (s) => s !== status,
            );
          } else {
            // 添加状态（去重）
            this.filters.selectedStatus = [
              ...new Set([...this.filters.selectedStatus, status]),
            ];
          }

          // 更新按钮视觉状态
          btnElement.classList.toggle("active", !isActive);

          // 同步筛选条件和显示
          this.config.currentPage = 1;
          this.updateTaskList();
          this.updateFilterStatus();
        });
      });

      // 查询按钮逻辑优化
      document.getElementById("search-btn").addEventListener("click", () => {
        const taskType = document.getElementById("task-type-filter").value;
        const taskName = document
          .getElementById("task-name-filter")
          .value.trim();

        // 修正校验逻辑：允许不输入任务名称
        if (
          !taskType &&
          !taskName &&
          this.filters.selectedStatus.length === 0
        ) {
          alert("请至少选择一个筛选条件");
          return;
        }

        this.filters.taskType = taskType;
        this.filters.taskName = taskName;
        this.config.currentPage = 1;
        this.updateTaskList();
      });
    },

    // 按钮状态更新
    updateButtonStatus() {
      document.querySelectorAll(".status-btn").forEach((btn) => {
        const status = btn.dataset.status;
        btn.classList.toggle(
          "active",
          this.filters.selectedStatus.includes(status),
        );
        // 添加样式变化
        if (btn.classList.contains("active")) {
          btn.style.backgroundColor = "#3b82f6";
          btn.style.color = "white";
        } else {
          btn.style.backgroundColor = "";
          btn.style.color = "";
        }
      });
    },

    // 筛选状态更新
    updateFilterStatus() {
      const statusMapping = {
        pending: "等待中",
        processing: "进行中",
        checking: "校验中",
        completed: "已完成",
        failed: "失败",
      };

      const activeFilters = this.filters.selectedStatus
        .map((s) => `状态：${statusMapping[s]}`)
        .concat(
          this.filters.taskType
            ? `类型：${this.taskTypes[this.filters.taskType]}`
            : "",
          this.filters.taskName ? `名称包含：${this.filters.taskName}` : "",
        )
        .filter(Boolean);

      document.getElementById("active-filters").textContent =
        activeFilters.join(" | ") || "无筛选条件";

      // 同步更新按钮样式
      this.updateButtonStatus();
    },

    // 获取任务列表
    async fetchTasks() {
      try {
        const params = new URLSearchParams();

        // 多状态筛选参数
        this.filters.selectedStatus.forEach((status) => {
          params.append("status", status);
        });

        if (this.filters.taskType) {
          params.append("task_type", this.filters.taskType);
        }
        if (this.filters.taskName) {
          params.append("name", this.filters.taskName);
        }

        // 添加分页参数
        params.append("page", this.config.currentPage);
        params.append("page_size", this.config.tasksPerPage);

        const response = await fetch(`/api/tasks/?${params.toString()}`);
        if (!response.ok) throw new Error(`请求失败: ${response.status}`);
        return await response.json();
      } catch (error) {
        console.error("获取任务失败:", error);
        throw error; // 抛出错误供上层处理
      }
    },

    formatDateTime(isoString) {
      const date = new Date(isoString);
      const year = date.getFullYear();
      const month = (date.getMonth() + 1).toString().padStart(2, "0");
      const day = date.getDate().toString().padStart(2, "0");
      const hour = date.getHours().toString().padStart(2, "0");
      const minute = date.getMinutes().toString().padStart(2, "0");
      return `${year}-${month}-${day} ${hour}:${minute}`;
    },

    // 创建任务卡片
    async createTaskCard(task) {
      console.log("创建任务卡片:", task.id, "is_favorite:", task.is_favorite);

      const card = document.createElement("div");
      card.className = `task-card p-4 border border-gray-100 relative`; // 保持基础卡片样式
      card.dataset.taskId = task.id;

      // 根据任务类型，定义用于任务类型文字标签的Tailwind类（在UI上区分不同标注任务）
      let taskTypeBgClass = "";
      let taskTypeTextColor = "text-gray-700"; // 默认颜色
      let taskTypeBorderColor = "border-gray-300"; // 默认边框色

      if (task.task_type === "image-classification") {
        taskTypeBgClass = "bg-blue-100"; // 浅蓝色背景
        taskTypeTextColor = "text-blue-700"; // 蓝色文字
        taskTypeBorderColor = "border-blue-300"; // 蓝色边框
      } else if (task.task_type === "text-classification") {
        taskTypeBgClass = "bg-green-100"; // 浅绿色背景
        taskTypeTextColor = "text-green-700"; // 绿色文字
        taskTypeBorderColor = "border-green-300"; // 绿色边框
      }

      // 直接使用后端返回的 task.is_favorite
      const isFavorite = task.is_favorite;
      console.log(`创建卡片 ${task.id}, 收藏状态: ${isFavorite}`); // 添加调试日志

      // 修复星标显示逻辑
      const starSymbol = isFavorite ? "★" : "☆"; // 实心/空心星
      const starClass = isFavorite ? "is-favorite" : ""; // 添加收藏状态类，依赖于CSS类控制颜色
      const starTitle = isFavorite ? "取消收藏" : "收藏此任务";

      const updateTimeHTML =
        (task.status === "completed" || task.status === "checking") &&
        task.updated_at
          ? `<div class="text-xs text-gray-400 mt-2">最后更新时间：${this.formatDateTime(task.updated_at)}</div>`
          : "";

      card.innerHTML = `
                <div class="flex items-start mb-4">
                    <div class="status-indicator ${this.statusColors[task.status]}
                         absolute top-3 right-3 w-3 h-3 rounded-full"></div>
                    <div class="favorite-star absolute top-3 left-3 text-lg cursor-pointer select-none ${starClass}"
                         title="${starTitle}"
                         data-task-id="${task.id}"
                         style="user-select: none; -webkit-user-select: none;">
                        ${starSymbol}
                    </div>
                    <div class="ml-6"> <!-- 添加左边距为星标留出空间 -->
                        <h3 class="text-lg font-semibold">${task.name}</h3>
                        <span class="inline-block px-2 py-0.5 rounded text-xs font-medium border
                                     ${taskTypeBgClass} ${taskTypeTextColor} ${taskTypeBorderColor}">
                            ${this.taskTypes[task.task_type]}
                        </span>
                        ${updateTimeHTML}
                    </div>
                </div>

                <div class="description-container text-sm text-gray-600 mb-4" style="min-height: 48px;">
                    <span class="font-semibold">任务描述：</span>${task.description ? `<span>${task.description}</span>` : "无"}
                </div>

                <div class="flex items-center justify-between text-sm">
                    <div class="flex-1 mr-4">
                        <div class="progress-bar bg-gray-200 rounded-full h-2">
                            <div class="progress-fill ${this.statusColors[task.status] || "bg-gray-400"}
                                h-2 rounded-full transition-all duration-500"
                                style="width: ${task.progress}%"></div>
                        </div>
                        <div class="flex justify-between text-xs mt-1">
                            <span class="progress-status-text">${this.statusTexts[task.status] || "未知"}</span>
                            <span class="progress-percent">${Number(task.progress || 0).toFixed(1)}%</span>
                        </div>
                    </div>

                    <div class="task-actions space-x-2">
                        <button class="detail-btn px-3 py-1 bg-blue-100 text-blue-600 rounded-md hover:bg-blue-200"
                                data-task-id="${task.id}"
                                data-task-type="${task.task_type}">
                            详情
                        </button>
                        <button class="delete-btn px-3 py-1 bg-red-100 text-red-600 rounded-md hover:bg-red-200"
                                data-task-id="${task.id}">
                            删除
                        </button>
                    </div>
                </div>
            `;

      // 启动进度轮询
      if (["pending", "processing", "checking"].includes(task.status)) {
        await this.startProgressPolling(task.id);
      }

      console.log("任务卡片创建完成:", task.id);
      return card;
    },

    // 启动进度轮询以更新进度条 (保持不变)
    async startProgressPolling(taskId) {
      let lastProgress = -1;
      let lastStatus = "";

      console.log(`开始轮询任务 ${taskId} 的进度`);

      const intervalId = setInterval(async () => {
        const card = document.querySelector(
          `.task-card[data-task-id="${taskId}"]`,
        );

        if (!card || !document.body.contains(card)) {
          console.log(`任务卡片 ${taskId} 已从DOM中移除，停止轮询`);
          clearInterval(intervalId);
          this.config.pollingIntervals.delete(taskId);
          return;
        }

        const elements = {
          fill: card.querySelector(".progress-fill"),
          percent: card.querySelector(".progress-percent"),
          indicator: card.querySelector(".status-indicator"),
          statusText: card.querySelector(".progress-status-text"),
        };

        if (
          !elements.fill ||
          !elements.percent ||
          !elements.indicator ||
          !elements.statusText
        ) {
          console.warn(`任务卡片 ${taskId} 元素不完整，跳过此次更新`);
          return;
        }

        try {
          const response = await fetch(`/api/tasks/${taskId}/progress/`);
          if (!response.ok) {
            if (response.status === 404) {
              console.warn(`任务 ${taskId} 不存在，停止轮询`);
              clearInterval(intervalId);
              this.config.pollingIntervals.delete(taskId);
              return;
            } else {
              console.error(`任务 ${taskId} 进度请求失败: ${response.status}`);
              return;
            }
          }

          const data = await response.json();
          const progressChanged = Math.abs(data.progress - lastProgress) > 0.1;
          const statusChanged = data.status !== lastStatus;

          if (!progressChanged && !statusChanged) {
            return;
          }

          console.log(
            `[任务 ${taskId}] 状态更新: ${data.progress.toFixed(1)}% (${data.status})`,
          );

          lastProgress = data.progress;
          lastStatus = data.status;

          if (progressChanged) {
            elements.percent.textContent = `${data.progress.toFixed(1)}%`;
            elements.fill.style.width = `${data.progress}%`;

            elements.fill.style.transition = "none";
            void elements.fill.offsetWidth;
            elements.fill.style.transition = "width 0.3s ease";
          }

          if (statusChanged) {
            const newColorClass =
              this.statusColors[data.status] || "bg-gray-400";
            elements.indicator.className = `status-indicator ${newColorClass} absolute top-3 right-3 w-3 h-3 rounded-full`;
            elements.statusText.textContent =
              this.statusTexts[data.status] || "未知";
            elements.fill.className = `progress-fill ${newColorClass} h-2 rounded-full transition-all duration-500`;

            if (["completed", "failed"].includes(data.status)) {
              console.log(
                `[任务 ${taskId}] 任务已结束，状态: ${data.status}，停止轮询`,
              );
              clearInterval(intervalId);
              this.config.pollingIntervals.delete(taskId);

              setTimeout(() => {
                this.updateTaskList();
              }, 1500);
            }
          }
        } catch (error) {
          console.error(`任务 ${taskId} 进度轮询失败:`, error);
        }
      }, 1500);

      this.config.pollingIntervals.set(taskId, intervalId);
    },

    async updateTaskList() {
      const taskList = document.getElementById("project-list");
      taskList.innerHTML =
        '<div class="col-span-3 text-center py-8">加载中...</div>';

      try {
        const { results, total } = await this.fetchTasks();

        // 分页计算
        this.config.totalPages = Math.ceil(total / this.config.tasksPerPage);

        // 清空列表
        taskList.innerHTML = "";

        if (results.length === 0) {
          taskList.innerHTML =
            '<div class="col-span-3 text-center py-8">没有找到相关任务</div>';
          return;
        }

        // 创建所有卡片
        // 注意：由于后端已经排好序，这里直接按顺序创建并append即可
        for (const task of results) {
          const card = await this.createTaskCard(task);
          taskList.appendChild(card);
        }

        // 更新分页和绑定事件
        this.updatePagination();
        this.bindDynamicEvents();
        this.updateFilterStatus();
      } catch (error) {
        taskList.innerHTML = `
                    <div class="col-span-3 text-center py-8 text-red-500">
                        加载失败: ${error.message}
                    </div>
                `;
      }
    },

    updatePagination() {
      document.getElementById("page-info").textContent =
        `第 ${this.config.currentPage} 页，共 ${this.config.totalPages} 页`;

      document.getElementById("prev-page").disabled =
        this.config.currentPage === 1;

      document.getElementById("next-page").disabled =
        this.config.currentPage === this.config.totalPages;
    },

    bindDynamicEvents() {
      // 移除旧的事件监听器
      if (this.globalClickHandler) {
        document.removeEventListener("click", this.globalClickHandler);
      }
      // 创建全局点击处理函数
      this.globalClickHandler = async (e) => {
        console.log("全局点击事件触发:", e.target);
        console.log("点击的元素类名:", e.target.className);
        console.log("点击的元素标签:", e.target.tagName);

        // 处理收藏按钮点击 - 增强调试
        const favoriteElement = e.target.closest(".favorite-star");
        console.log("找到的收藏元素:", favoriteElement);

        if (favoriteElement) {
          console.log("收藏元素被点击");
          e.preventDefault();
          e.stopPropagation();

          const taskId = favoriteElement.dataset.taskId;
          console.log("从dataset获取的taskId:", taskId);
          console.log("收藏元素的所有dataset:", favoriteElement.dataset);

          if (!taskId) {
            console.error("未找到任务ID");
            this.showToast("操作失败：未找到任务ID", "error");
            return;
          }

          console.log("开始执行收藏操作，任务ID:", taskId);

          try {
            await this.toggleFavorite(taskId, favoriteElement);
          } catch (error) {
            console.error("收藏操作失败:", error);
          }
          return;
        }

        // 处理删除按钮点击
        const deleteBtn = e.target.closest(".delete-btn");
        if (deleteBtn) {
          const taskId = deleteBtn.dataset.taskId;
          console.log("删除按钮点击，任务ID:", taskId);

          if (confirm("确定删除此任务吗？")) {
            deleteBtn.innerHTML =
              '<i class="fas fa-spinner fa-spin"></i> 删除中...';
            deleteBtn.disabled = true;

            try {
              const response = await fetch(`/api/tasks/${taskId}/`, {
                method: "DELETE",
                headers: { "X-CSRFToken": this.getCSRFToken() },
              });

              const result = await response.json();
              if (response.ok) {
                this.showToast("删除成功", "success");
                this.updateTaskList();
              } else {
                this.showToast(
                  `删除失败: ${result.error || "未知错误"}`,
                  "error",
                );
              }
            } catch (error) {
              this.showToast("网络请求异常，请检查控制台", "error");
              console.error("删除失败:", error);
            } finally {
              deleteBtn.innerHTML = "删除";
              deleteBtn.disabled = false;
            }
          }
          return;
        }

        // 处理详情按钮点击
        const detailBtn = e.target.closest(".detail-btn");
        if (detailBtn) {
          const taskId = detailBtn.dataset.taskId;
          const taskType = detailBtn.dataset.taskType;
          console.log("详情按钮点击，任务ID:", taskId, "任务类型:", taskType);

          // 在当前页面重定向到详情页，不再打开新标签
          window.location.href = `/tasks/${taskId}/detail/?type=${taskType}`;
          return;
        }
      };
      // 添加全局事件监听器
      document.addEventListener("click", this.globalClickHandler);
    },
    // 5. 添加收藏状态变化监听器
    bindMessageListener() {
      window.addEventListener("message", (event) => {
        if (event.data.type === "favoriteChanged") {
          console.log("收到收藏状态变化消息:", event.data);
          // 更新当前页面的收藏显示
          this.updateTaskFavoriteDisplay(
            event.data.taskId,
            event.data.isFavorite,
          );
        }
      });
    },

    // 6. 添加更新任务收藏显示的函数
    updateTaskFavoriteDisplay(taskId, isFavorite) {
      const card = document.querySelector(`[data-task-id="${taskId}"]`);
      if (card) {
        const starElement = card.querySelector(".favorite-star");
        if (starElement) {
          this.updateStarDisplay(starElement, isFavorite); // 复用更新显示逻辑
        }
      }
    },
    getCSRFToken() {
      const cookieValue = document.cookie
        .split("; ")
        .find((row) => row.startsWith("csrftoken="))
        ?.split("=")[1];
      return cookieValue || "";
    },
  };

  // 分页控制
  document.getElementById("prev-page").addEventListener("click", () => {
    if (TaskManager.config.currentPage > 1) {
      TaskManager.config.currentPage--;
      TaskManager.updateTaskList();
    }
  });

  document.getElementById("next-page").addEventListener("click", () => {
    if (TaskManager.config.currentPage < TaskManager.config.totalPages) {
      TaskManager.config.currentPage++;
      TaskManager.updateTaskList();
    }
  });

  // 初始化加载与事件绑定
  TaskManager.bindFilterEvents();
  TaskManager.updateTaskList();

  // 初始化消息监听器
  TaskManager.bindMessageListener();

  // 清理轮询
  window.addEventListener("beforeunload", () => {
    TaskManager.config.pollingIntervals.forEach((interval, taskId) => {
      clearInterval(interval);
    });
  });

  // DOM加载完成日志
  console.log("管理页面DOM加载完成，TaskManager已初始化");
});
