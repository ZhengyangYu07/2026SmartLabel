/**
 * 目标检测标注页（LabelMe 风格）主脚本。
 * 负责：标注画布上的矩形/多边形/点/线绘制与拖拽编辑、标注框的选择/改名/接受/拒绝、
 * 撤销重做、缩放与居中、图片列表与分页加载、标签联想、标注结果保存与“完成当前图片”流转。
 * 依赖页面提供的 window.detectionTaskId / detectionTaskLabels / detectionViewMode / currentImageItem 等全局变量。
 */
(() => {
  const taskId = window.detectionTaskId;
  // 界面文案表（用 Unicode 转义书写，避免编码问题）
  const TXT = {
    unnamed: "\u672a\u547d\u540d",
    noImage: "\u8bf7\u9009\u62e9\u56fe\u7247",
    noShape: "\u6682\u65e0\u6807\u6ce8\u6846",
    noLabel: "\u6682\u65e0\u6807\u7b7e",
    loading: "\u52a0\u8f7d\u4e2d...",
    loadFail: "\u6570\u636e\u52a0\u8f7d\u5931\u8d25",
    saveFirst: "\u8bf7\u5148\u9009\u62e9\u56fe\u7247",
    saved: "\u6807\u6ce8\u5df2\u4fdd\u5b58",
    completed: "\u5f53\u524d\u56fe\u7247\u5df2\u6807\u6ce8\u5b8c\u6210",
    saveFail: "\u4fdd\u5b58\u5931\u8d25",
    noUndo: "\u6ca1\u6709\u53ef\u64a4\u9500\u7684\u64cd\u4f5c",
    chooseLabel: "\u9009\u62e9\u6807\u7b7e",
    verified: "\u5df2\u6821\u9a8c",
    unverified: "\u672a\u6821\u9a8c",
    shape: "\u6807\u6ce8\u6846",
    shapeCount: "\u4e2a\u6807\u6ce8\u6846",
    uncertainty: "\u4e0d\u786e\u5b9a\u6027",
    confidence: "\u7f6e\u4fe1\u5ea6",
    image: "\u56fe\u7247",
    pagePrev: "\u4e0a\u4e00\u9875",
    pageNext: "\u4e0b\u4e00\u9875",
    rectangle: "\u77e9\u5f62",
    polygon: "\u591a\u8fb9\u5f62",
    point: "\u70b9",
    line: "\u7ebf",
    pending: "\u5f85\u5904\u7406",
    accepted: "\u5df2\u63a5\u53d7",
    rejected: "\u5df2\u62d2\u7edd",
    modified: "\u5df2\u4fee\u6539",
    manual: "\u4eba\u5de5\u65b0\u589e",
    model: "\u6a21\u578b\u9884\u6d4b",
    precheck: "\u9884\u6807\u6ce8\u6821\u9a8c",
    activeLearning: "\u4e3b\u52a8\u5b66\u4e60",
    high: "\u9ad8",
    medium: "\u4e2d",
    low: "\u4f4e",
    hideAnnotations: "\u9690\u85cf\u6807\u6ce8\u6846",
    showAnnotations: "\u663e\u793a\u6807\u6ce8\u6846",
    acceptAllWarn: "\u5f53\u524d\u662f\u4e3b\u52a8\u5b66\u4e60\u6837\u672c\uff0c\u4e0d\u5efa\u8bae\u5168\u90e8\u63a5\u53d7\uff0c\u8bf7\u9010\u6846\u5904\u7406\u3002",
  };

  // 低置信度阈值：低于此值的框会被高亮提示
  const LOW_CONFIDENCE = 0.6;
  // 主动学习样本的不确定性阈值
  const ACTIVE_UNCERTAINTY = 0.5;
  // 占位标签名（无真实标签时使用）
  const PLACEHOLDER_LABEL = "obj";

  // 缓存页面上的各类 DOM 元素引用
  const container = document.getElementById("draw-canvas-container");
  const imageEl = document.getElementById("detection-image");
  const selectBtn = document.getElementById("select-mode-btn");
  const deleteBtn = document.getElementById("delete-box-btn");
  const saveBtn = document.getElementById("save-boxes-btn");
  const completeImageBtn = document.getElementById("complete-image-btn");
  const undoBtn = document.getElementById("undo-btn");
  const acceptAllBtn = document.getElementById("accept-all-btn");
  const finishPolygonBtn = document.getElementById("finish-polygon-btn");
  const modeBadge = document.getElementById("mode-badge");
  const labelSelect = document.getElementById("box-label-select");
  const statusFilter = document.getElementById("status-filter");
  const sortSelect = document.getElementById("sort-filter");
  const imageList = document.getElementById("detection-image-list");
  const objectList = document.getElementById("object-list");
  const labelListPanel = document.getElementById("label-list-panel");
  const pager = document.getElementById("detection-pager");
  const currentName = document.getElementById("current-image-name");
  const currentMeta = document.getElementById("current-image-meta");
  const queueMeta = document.getElementById("queue-meta");
  const boxCountChip = document.getElementById("box-count-chip");
  const sampleUncertainty = document.getElementById("sample-uncertainty");
  const sampleStrategy = document.getElementById("sample-strategy");
  const sampleModelVersion = document.getElementById("sample-model-version");
  const samplePriority = document.getElementById("sample-priority");
  const prevImageBtn = document.getElementById("prev-image-btn");
  const nextImageBtn = document.getElementById("next-image-btn");
  const zoomInBtn = document.getElementById("zoom-in-btn");
  const zoomOutBtn = document.getElementById("zoom-out-btn");
  const fitWindowBtn = document.getElementById("fit-window-btn");
  const toggleAnnotationsBtn = document.getElementById("toggle-annotations-btn");
  const toolButtons = [...document.querySelectorAll("[data-tool]")];
  const tabButtons = [...document.querySelectorAll(".det-tab")];

  // 运行状态：当前模式/工具、形状集合、绘制与拖拽临时状态、分页、缩放、撤销栈等
  let mode = "draw";
  let activeTool = "rectangle";
  let shapes = [];
  let drawing = false;
  let dragStart = null;
  let activeShape = null;
  let currentPage = 1;
  let currentItems = [];
  let currentItemIndex = -1;
  let currentTotalCount = 0;
  let currentTotalPages = 1;
  let loadedLabels = [];
  let zoomScale = 1;
  let draftPoints = [];
  let draftEl = null;
  let svg = null;
  let rectEdit = null;
  let undoStack = [];
  let hasUnsavedChanges = false;
  let baseImageSize = { width: 0, height: 0 };
  let annotationsVisible = true;

  // 将数值限制在 [0,1] 区间（坐标统一用归一化值存储）
  function clamp01(value) {
    return Math.max(0, Math.min(1, Number(value) || 0));
  }

  // 数字格式化：非有限值显示为 "-"，否则保留指定小数位
  function fmt(value, digits = 3) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "-";
  }

  // HTML 转义，防止把标签名等内容直接拼进 innerHTML 时被注入
  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  // 取形状的置信度，非法则返回 null
  function confidenceValue(shape) {
    const value = Number(shape.confidence);
    return Number.isFinite(value) ? value : null;
  }

  // 处理决策状态文案（pending/accepted/rejected/modified）
  function decisionText(decision) {
    return TXT[decision] || TXT.pending;
  }

  // 标注来源文案（人工新增 / 模型预测）
  function sourceText(source) {
    return source === "manual" ? TXT.manual : TXT.model;
  }

  // 形状类型文案（矩形/多边形/点/线）
  function shapeName(shapeType) {
    return TXT[shapeType] || TXT.shape;
  }

  // 读取当前图片的不确定性分数
  function currentUncertainty() {
    return Number(window.currentImageItem?.uncertainty_score || 0);
  }

  // 判断当前图片是否属于主动学习样本（不确定性达标）
  function isActiveLearningSample() {
    return currentUncertainty() >= ACTIVE_UNCERTAINTY;
  }

  // 按分数映射优先级文案（高/中/低）
  function priorityText(score) {
    if (score >= 0.7) return TXT.high;
    if (score >= 0.35) return TXT.medium;
    return TXT.low;
  }

  // 切换绘制/选择模式，并同步光标样式与按钮高亮
  function setMode(nextMode) {
    mode = nextMode;
    selectBtn?.classList.toggle("active", mode === "select");
    if (container) container.style.cursor = mode === "select" ? "default" : "crosshair";
  }

  // 切换标注工具（矩形/多边形/点/线），并清空未完成的绘制草稿
  function setTool(nextTool) {
    activeTool = nextTool;
    setMode("draw");
    draftPoints = [];
    removeDraft();
    updateFinishPolygonState();
    updateUndoState();
    toolButtons.forEach((button) => {
      button.classList.toggle("active", button.dataset.tool === nextTool);
    });
  }

  // 设置缩放比例（限制最小 0.1），按基准尺寸重算图片显示尺寸并重绘标注
  function setZoom(nextZoom) {
    const requestedZoom = Number(nextZoom);
    zoomScale = Math.max(0.1, Number.isFinite(requestedZoom) ? requestedZoom : 1);
    if (baseImageSize.width && baseImageSize.height) {
      imageEl.style.maxWidth = "none";
      imageEl.style.maxHeight = "none";
      imageEl.style.width = `${baseImageSize.width * zoomScale}px`;
      imageEl.style.height = `${baseImageSize.height * zoomScale}px`;
    }
    rerenderShapes();
    updateCanvasOverflow();
    centerCanvas();
  }

  // 显示/隐藏所有标注框，并更新按钮文案
  function setAnnotationsVisible(visible) {
    annotationsVisible = visible;
    container?.classList.toggle("annotations-hidden", !annotationsVisible);
    if (toggleAnnotationsBtn) {
      toggleAnnotationsBtn.textContent = annotationsVisible ? TXT.hideAnnotations : TXT.showAnnotations;
    }
  }

  // 根据画布是否超出视口，切换滚动容器的横向/纵向滚动标记
  function updateCanvasOverflow() {
    const viewport = container?.closest(".lm-canvas-area");
    if (!viewport) return;
    const padding = 48;
    viewport.classList.toggle("canvas-overflow-x", container.offsetWidth + padding > viewport.clientWidth);
    viewport.classList.toggle("canvas-overflow-y", container.offsetHeight + padding > viewport.clientHeight);
  }

  // 将画布滚动到视口中央
  function centerCanvas() {
    const viewport = container?.closest(".lm-canvas-area");
    if (!viewport) return;
    window.requestAnimationFrame(() => {
      viewport.scrollLeft = Math.max(0, (viewport.scrollWidth - viewport.clientWidth) / 2);
      viewport.scrollTop = Math.max(0, (viewport.scrollHeight - viewport.clientHeight) / 2);
    });
  }

  // 由标签文本哈希出一个稳定颜色，保证同一标签颜色一致
  function colorForLabel(label) {
    const text = label || TXT.unnamed;
    let hash = 0;
    for (let i = 0; i < text.length; i += 1) {
      hash = (hash * 31 + text.charCodeAt(i)) >>> 0;
    }
    return `hsl(${hash % 360}, 72%, 45%)`;
  }

  // 按决策/来源/置信度决定标注框颜色
  function colorForShape(shape) {
    const conf = confidenceValue(shape);
    if (shape.decision === "rejected") return "#dc2626";
    if (shape.decision === "accepted") return "#16a34a";
    if (shape.decision === "manual" || shape.source === "manual" || shape.decision === "modified") return "#eab308";
    if (conf !== null && conf < LOW_CONFIDENCE) return "#f97316";
    return "#2563eb";
  }

  // 读取图片当前显示尺寸（像素）
  function imageSize() {
    return {
      width: imageEl.clientWidth,
      height: imageEl.clientHeight,
    };
  }

  // 归一化坐标 -> 图片上的像素坐标
  function pointToPx(point) {
    const normalized = Array.isArray(point) ? point : [point?.x, point?.y];
    const size = imageSize();
    return {
      x: clamp01(normalized[0]) * size.width,
      y: clamp01(normalized[1]) * size.height,
    };
  }

  // 鼠标事件的屏幕坐标 -> 图片归一化坐标（用于绘制）
  function toRelativeCoords(clientX, clientY) {
    const rect = imageEl.getBoundingClientRect();
    return {
      x: clamp01((clientX - rect.left) / rect.width),
      y: clamp01((clientY - rect.top) / rect.height),
    };
  }

  // 懒创建承载 SVG 形状（多边形/线/点）的图层
  function ensureSvg() {
    if (svg) return svg;
    svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("shape-svg");
    container.appendChild(svg);
    return svg;
  }

  // 让 SVG 覆盖层尺寸与当前图片显示尺寸对齐
  function sizeOverlay() {
    const size = imageSize();
    const layer = ensureSvg();
    layer.setAttribute("width", String(size.width));
    layer.setAttribute("height", String(size.height));
    layer.style.width = `${size.width}px`;
    layer.style.height = `${size.height}px`;
  }

  // 移除正在绘制中的临时草稿（如多边形预览线）
  function removeDraft() {
    draftEl?.remove();
    draftEl = null;
  }

  // 清空全部标注框与 SVG 图层，重置选中/草稿状态
  function clearOverlays() {
    shapes.forEach((shape) => {
      shape.el?.remove();
      shape.labelEl?.remove();
    });
    shapes = [];
    activeShape = null;
    draftPoints = [];
    removeDraft();
    svg?.remove();
    svg = null;
    updateShapePanel();
  }

  // 由点集计算包围盒（归一化坐标下的 min/max）
  function boundsFromPoints(points) {
    const xs = points.map((point) => clamp01(point[0]));
    const ys = points.map((point) => clamp01(point[1]));
    return {
      minX: Math.min(...xs),
      minY: Math.min(...ys),
      maxX: Math.max(...xs),
      maxY: Math.max(...ys),
    };
  }

  // 计算标签文字的锚点位置（包围盒左上角）
  function anchorPoint(shape) {
    if (shape.points?.length) {
      const bounds = boundsFromPoints(shape.points);
      return [bounds.minX, bounds.minY];
    }
    return [0, 0];
  }

  // 拼接标签文字（含置信度）
  function labelText(shape) {
    const conf = confidenceValue(shape);
    const label = shape.label || TXT.unnamed;
    return conf === null ? label : `${label} ${fmt(conf, 2)}`;
  }

  // 定位并刷新形状标签气泡的位置、颜色与文字
  function positionLabel(shape) {
    if (!shape.labelEl) return;
    const point = pointToPx(anchorPoint(shape));
    shape.labelEl.style.left = `${point.x}px`;
    shape.labelEl.style.top = `${point.y}px`;
    shape.labelEl.style.background = colorForShape(shape);
    shape.labelEl.textContent = labelText(shape);
  }

  // 为形状创建标签气泡 DOM
  function createLabel(shape) {
    const labelEl = document.createElement("div");
    labelEl.className = "bbox-label";
    container.appendChild(labelEl);
    shape.labelEl = labelEl;
    positionLabel(shape);
  }

  // 给矩形元素加上来源/状态/低置信/选中等样式类
  function applyRectangleClasses(shape) {
    if (!shape.el) return;
    shape.el.className = "bbox";
    shape.el.classList.add(`bbox-source-${shape.source || "model"}`);
    shape.el.classList.add(`bbox-status-${shape.decision || "pending"}`);
    if (confidenceValue(shape) !== null && confidenceValue(shape) < LOW_CONFIDENCE) {
      shape.el.classList.add("bbox-low-confidence");
    }
    if (shape === activeShape) shape.el.classList.add("bbox-active");
  }

  // 渲染矩形：创建框体、四角拖拽手柄，并绑定选中/拖拽事件
  function renderRectangle(shape) {
    const el = document.createElement("div");
    el.addEventListener("mousedown", (event) => {
      event.stopPropagation();
    });
    el.addEventListener("click", (event) => {
      event.stopPropagation();
      selectShape(shape);
    });
    ["nw", "ne", "sw", "se"].forEach((handle) => {
      const dot = document.createElement("span");
      dot.className = "bbox-handle";
      dot.dataset.handle = handle;
      dot.addEventListener("mousedown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        selectShape(shape);
        const bounds = boundsFromPoints(shape.points);
        pushUndo();
        rectEdit = { shape, handle, bounds };
      });
      el.appendChild(dot);
    });
    container.appendChild(el);
    shape.el = el;
    createLabel(shape);
    positionShape(shape);
  }

  // 将点集转换为 SVG polyline/polygon 的 "x,y" 坐标串
  function svgPoints(points) {
    return points
      .map((point) => {
        const px = pointToPx(point);
        return `${px.x},${px.y}`;
      })
      .join(" ");
  }

  // 渲染 SVG 形状（点/线/多边形）并绑定点击选中
  function renderSvgShape(shape) {
    const layer = ensureSvg();
    let el;
    if (shape.shape_type === "point") {
      el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      el.classList.add("shape-point");
      el.setAttribute("r", "5");
      el.setAttribute("stroke", "#ffffff");
      el.setAttribute("stroke-width", "2");
    } else if (shape.shape_type === "line") {
      el = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      el.classList.add("shape-line");
      el.setAttribute("fill", "none");
      el.setAttribute("stroke-width", "3");
      el.setAttribute("stroke-linecap", "round");
      el.setAttribute("stroke-linejoin", "round");
    } else {
      el = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      el.classList.add("shape-polygon");
      el.setAttribute("stroke-width", "2");
    }
    el.addEventListener("click", (event) => {
      event.stopPropagation();
      selectShape(shape);
    });
    layer.appendChild(el);
    shape.el = el;
    createLabel(shape);
    positionShape(shape);
  }

  // 按类型把形状几何与颜色同步到其 DOM/SVG 元素
  function positionShape(shape) {
    if (!shape.el || !imageEl.clientWidth || !imageEl.clientHeight) return;
    sizeOverlay();
    const color = colorForShape(shape);
    if (shape.shape_type === "rectangle") {
      const bounds = boundsFromPoints(shape.points);
      const size = imageSize();
      shape.el.style.left = `${bounds.minX * size.width}px`;
      shape.el.style.top = `${bounds.minY * size.height}px`;
      shape.el.style.width = `${(bounds.maxX - bounds.minX) * size.width}px`;
      shape.el.style.height = `${(bounds.maxY - bounds.minY) * size.height}px`;
      shape.el.style.borderColor = color;
      applyRectangleClasses(shape);
    } else if (shape.shape_type === "point") {
      const point = pointToPx(shape.points[0] || [0, 0]);
      shape.el.setAttribute("cx", String(point.x));
      shape.el.setAttribute("cy", String(point.y));
      shape.el.setAttribute("fill", color);
      shape.el.setAttribute("stroke", shape === activeShape ? "#ffcc00" : "#ffffff");
    } else {
      shape.el.setAttribute("points", svgPoints(shape.points));
      shape.el.setAttribute("stroke", color);
      shape.el.classList.toggle("shape-active", shape === activeShape);
      if (shape.shape_type === "polygon") {
        shape.el.setAttribute("fill", color);
        shape.el.setAttribute("fill-opacity", "0.06");
        shape.el.setAttribute("stroke-width", shape === activeShape ? "3" : "2");
      }
    }
    positionLabel(shape);
  }

  // 按形状类型分发渲染（矩形走 DOM，其余走 SVG）
  function renderShape(shape) {
    if (shape.shape_type === "rectangle") renderRectangle(shape);
    else renderSvgShape(shape);
  }

  // 重绘全部形状（缩放/尺寸变化后调用）
  function rerenderShapes() {
    sizeOverlay();
    shapes.forEach(positionShape);
    renderDraft();
  }

  // 选中某个形状（或取消选中），并同步标签下拉与对象面板
  function selectShape(shape) {
    activeShape = shape || null;
    shapes.forEach(positionShape);
    if (labelSelect) labelSelect.value = activeShape?.label || "";
    updateShapePanel();
  }

  // 设置形状的决策状态（接受/拒绝/修改），默认记入撤销栈
  function setShapeDecision(shape, decision, record = true) {
    if (!shape) return;
    if (record) pushUndo();
    shape.decision = decision;
    if (decision === "modified") shape.source = "manual";
    positionShape(shape);
    updateShapePanel();
  }

  // 将形状标记为“已修改/人工来源”（模型框被人工编辑后调用）
  function markShapeModified(shape) {
    if (!shape) return;
    shape.source = "manual";
    shape.decision = shape.decision === "manual" ? "manual" : "modified";
  }

  // 修改形状标签名，并同步标签候选与下拉选项
  function renameShape(shape, label, record = true) {
    if (!shape) return;
    if (record) pushUndo();
    shape.label = label || "";
    rememberLabel(shape.label);
    refreshLabelSelectOptions(shape.label);
    markShapeModified(shape);
    if (labelSelect && shape === activeShape) labelSelect.value = shape.label;
    positionShape(shape);
    updateShapePanel();
  }

  // 删除当前选中形状；若正在画多边形则先清除草稿点
  function removeActiveShape() {
    if (draftPoints.length) {
      draftPoints = [];
      removeDraft();
      updateFinishPolygonState();
      updateUndoState();
      return;
    }
    if (!activeShape) return;
    pushUndo();
    const target = activeShape;
    shapes = shapes.filter((shape) => shape !== target);
    target.el?.remove();
    target.labelEl?.remove();
    activeShape = null;
    updateShapePanel();
  }

  // 后端标注数据 -> 前端内部形状对象（统一坐标为两点包围盒）
  function annotationToShape(annotation, item) {
    const shapeType = annotation.shape_type || "rectangle";
    const hasModelConfidence = annotation.confidence !== undefined && annotation.confidence !== null && annotation.confidence !== "";
    const source = annotation.source || (hasModelConfidence || item?.confidence !== "-1.000" ? "model" : "manual");
    let decision = annotation.decision || annotation.annotation_status || (source === "model" ? "pending" : "manual");
    const isManualVerifiedItem = item?.confidence === "-1.000" && (item?.raw_status === "verified" || item?.status === TXT.verified);
    if (isManualVerifiedItem && decision !== "rejected") decision = "accepted";
    const base = {
      shape_type: shapeType,
      label: annotation.label || "",
      confidence: hasModelConfidence ? annotation.confidence : undefined,
      source,
      decision,
    };
    if (shapeType !== "rectangle" && Array.isArray(annotation.points)) {
      return {
        ...base,
        points: annotation.points.map((point) => [clamp01(point[0]), clamp01(point[1])]),
      };
    }
    if (Array.isArray(annotation.points) && annotation.points.length >= 2) {
      const bounds = boundsFromPoints(annotation.points);
      return {
        ...base,
        shape_type: "rectangle",
        points: [[bounds.minX, bounds.minY], [bounds.maxX, bounds.maxY]],
      };
    }
    const w = clamp01(annotation.w);
    const h = clamp01(annotation.h);
    const centerX = clamp01(annotation.x);
    const centerY = clamp01(annotation.y);
    return {
      ...base,
      shape_type: "rectangle",
      points: [
        [clamp01(centerX - w / 2), clamp01(centerY - h / 2)],
        [clamp01(centerX + w / 2), clamp01(centerY + h / 2)],
      ],
    };
  }

  // 前端形状对象 -> 保存接口的 payload（矩形补充 x/y/w/h）
  function shapeToPayload(shape) {
    const payload = {
      shape_type: shape.shape_type,
      label: shape.label || "",
      source: shape.source || "manual",
      decision: shape.decision || "manual",
      points: shape.points.map((point) => [clamp01(point[0]), clamp01(point[1])]),
    };
    if (shape.confidence !== undefined) payload.confidence = shape.confidence;
    if (shape.shape_type === "rectangle") {
      const bounds = boundsFromPoints(payload.points);
      payload.x = bounds.minX;
      payload.y = bounds.minY;
      payload.w = bounds.maxX - bounds.minX;
      payload.h = bounds.maxY - bounds.minY;
    }
    return payload;
  }

  // 生成形状的轻量快照，用于撤销栈存储
  function shapeState(shape) {
    const state = {
      shape_type: shape.shape_type,
      label: shape.label || "",
      source: shape.source || "manual",
      decision: shape.decision || "manual",
      points: (shape.points || []).map((point) => [clamp01(point[0]), clamp01(point[1])]),
    };
    if (shape.confidence !== undefined) state.confidence = shape.confidence;
    return state;
  }

  // 刷新撤销按钮可用状态
  function updateUndoState() {
    if (undoBtn) undoBtn.disabled = undoStack.length === 0 && draftPoints.length === 0;
  }

  // 把当前形状集合快照压入撤销栈（最多保留 60 步）
  function pushUndo() {
    if (!window.currentImageItem) return;
    undoStack.push(shapes.map(shapeState));
    if (undoStack.length > 60) undoStack.shift();
    hasUnsavedChanges = true;
    updateUndoState();
  }

  // 由撤销快照重建全部形状并重新渲染
  function restoreShapes(snapshot) {
    shapes.forEach((shape) => {
      shape.el?.remove();
      shape.labelEl?.remove();
    });
    activeShape = null;
    draftPoints = [];
    removeDraft();
    svg?.remove();
    svg = null;
    shapes = snapshot.map((shape) => ({
      ...shape,
      points: (shape.points || []).map((point) => [clamp01(point[0]), clamp01(point[1])]),
    }));
    shapes.forEach(renderShape);
    rerenderShapes();
    selectShape(null);
    updateShapePanel();
  }

  // 撤销上一步：优先撤销多边形草稿点，否则弹出撤销栈快照
  function undoLastStep() {
    if (draftPoints.length) {
      draftPoints.pop();
      renderDraft();
      updateFinishPolygonState();
      updateUndoState();
      return;
    }
    const snapshot = undoStack.pop();
    if (!snapshot) {
      window.showMessage(TXT.noUndo, "warning");
      return;
    }
    restoreShapes(snapshot);
    hasUnsavedChanges = true;
    updateUndoState();
  }

  // 渲染左侧标签列表面板
  function renderLabelList() {
    if (!labelListPanel) return;
    if (!loadedLabels.length) {
      labelListPanel.innerHTML = `<div class="det-empty">${TXT.noLabel}</div>`;
      return;
    }
    labelListPanel.innerHTML = loadedLabels
      .map((label, index) => {
        const safeLabel = escapeHtml(label);
        const active = labelSelect?.value === label ? "active" : "";
        return `
          <button type="button" class="det-label-row ${active}" data-label="${safeLabel}">
            <span class="lm-color-dot" style="background:${colorForLabel(label)}"></span>
            <span>${index + 1}. ${safeLabel}</span>
          </button>`;
      })
      .join("");
    [...labelListPanel.querySelectorAll(".det-label-row")].forEach((row) => {
      row.addEventListener("click", () => {
        const label = row.dataset.label || "";
        if (labelSelect) labelSelect.value = label;
        if (activeShape) {
          pushUndo();
          activeShape.label = label;
          rememberLabel(label);
          if (activeShape.source === "model" && activeShape.decision !== "accepted") {
            activeShape.decision = "modified";
            activeShape.source = "manual";
          }
          positionShape(activeShape);
          updateShapePanel();
        }
        renderLabelList();
      });
    });
  }

  // 记录一个标签到候选列表（去重并排序）
  function rememberLabel(label) {
    const value = String(label || "").trim();
    if (!value || loadedLabels.includes(value)) return;
    loadedLabels.push(value);
    loadedLabels.sort((a, b) => a.localeCompare(b, "zh-Hans-CN"));
    removePlaceholderLabel();
  }

  // 是否已加载到真实标签（非占位标签）
  function hasRealLoadedLabel() {
    return loadedLabels.some((label) => label !== PLACEHOLDER_LABEL);
  }

  // 有真实标签时移除占位标签 "obj"
  function removePlaceholderLabel() {
    if (!hasRealLoadedLabel()) return;
    loadedLabels = loadedLabels.filter((label) => label !== PLACEHOLDER_LABEL);
  }

  // 清掉形状上残留的占位标签名
  function clearPlaceholderShapeLabels() {
    if (!hasRealLoadedLabel()) return;
    shapes.forEach((shape) => {
      if (shape.label === PLACEHOLDER_LABEL) shape.label = "";
    });
  }

  // 重建标签下拉框选项，尽量保留原选中值
  function refreshLabelSelectOptions(selectedValue = null) {
    if (!labelSelect) return;
    removePlaceholderLabel();
    const currentValue = selectedValue ?? labelSelect.value;
    labelSelect.innerHTML = `<option value="">${TXT.chooseLabel}</option>`;
    loadedLabels.forEach((label) => {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      labelSelect.appendChild(option);
    });
    if (currentValue && loadedLabels.includes(currentValue)) {
      labelSelect.value = currentValue;
    } else if (!currentValue && loadedLabels.length) {
      labelSelect.value = loadedLabels[0];
    }
    renderLabelList();
  }

  // 从形状集合中汇总标签并刷新下拉选项
  function collectLabelsFromShapes(shapeList) {
    shapeList.forEach((shape) => rememberLabel(shape.label));
    refreshLabelSelectOptions(labelSelect?.value || shapeList[0]?.label || null);
  }

  // 用任务预置标签初始化候选列表
  function seedTaskLabels() {
    if (!Array.isArray(window.detectionTaskLabels)) return;
    window.detectionTaskLabels.forEach(rememberLabel);
    refreshLabelSelectOptions();
  }

  // 按关键词过滤标签候选
  function matchingLabels(searchText) {
    const keyword = String(searchText || "").trim().toLowerCase();
    if (!keyword) return [...loadedLabels];
    return loadedLabels.filter((label) => label.toLowerCase().includes(keyword));
  }

  // 在对象行的标签输入框下渲染联想候选
  function renderInlineLabelSuggestions(input) {
    const index = input.dataset.labelInput;
    const list = objectList?.querySelector(`[data-label-suggestions="${index}"]`);
    if (!list) return;
    const matches = matchingLabels(input.value);
    if (!matches.length) {
      list.innerHTML = `<button type="button" class="det-label-suggestion" disabled>${TXT.noLabel}</button>`;
      list.classList.add("visible");
      return;
    }
    list.innerHTML = matches
      .map((label) => `<button type="button" class="det-label-suggestion" data-suggest-label="${escapeHtml(label)}">${escapeHtml(label)}</button>`)
      .join("");
    [...list.querySelectorAll("[data-suggest-label]")].forEach((button) => {
      button.addEventListener("mousedown", (event) => {
        event.preventDefault();
        const shape = shapes[Number(index)];
        const label = button.dataset.suggestLabel || "";
        input.value = label;
        selectShape(shape);
        renameShape(shape, label);
        list.classList.remove("visible");
      });
    });
    list.classList.add("visible");
  }

  // 隐藏标签联想候选列表
  function hideInlineLabelSuggestions(input) {
    const list = objectList?.querySelector(`[data-label-suggestions="${input.dataset.labelInput}"]`);
    list?.classList.remove("visible");
  }

  // 排序形状用于对象面板展示：低置信、待处理优先，再按置信度升序
  function sortedShapeEntries() {
    return shapes
      .map((shape, index) => ({ shape, index }))
      .sort((a, b) => {
        const ac = confidenceValue(a.shape);
        const bc = confidenceValue(b.shape);
        const aLow = ac !== null && ac < LOW_CONFIDENCE ? 0 : 1;
        const bLow = bc !== null && bc < LOW_CONFIDENCE ? 0 : 1;
        if (aLow !== bLow) return aLow - bLow;
        const aPending = a.shape.decision === "pending" ? 0 : 1;
        const bPending = b.shape.decision === "pending" ? 0 : 1;
        if (aPending !== bPending) return aPending - bPending;
        return (ac ?? 1) - (bc ?? 1);
      });
  }

  // 渲染右侧对象面板：每个标注框的名称/置信度/状态/接受-修改-拒绝按钮
  function updateShapePanel() {
    if (boxCountChip) boxCountChip.textContent = `${shapes.length} ${TXT.shapeCount}`;
    if (!objectList) return;
    if (!shapes.length) {
      objectList.innerHTML = `<div class="det-empty">${TXT.noShape}</div>`;
    } else {
      objectList.innerHTML = sortedShapeEntries()
        .map(({ shape, index }) => {
          const active = shape === activeShape ? "active" : "";
          const label = shape.label || TXT.unnamed;
          const conf = confidenceValue(shape);
          const confWidth = conf === null ? 0 : Math.round(clamp01(conf) * 100);
          const low = conf !== null && conf < LOW_CONFIDENCE ? "hot" : "ok";
          const confChip = conf === null
            ? ""
            : `<span class="det-chip ${low}">${TXT.confidence} ${fmt(conf)}</span>`;
          return `
            <div class="det-object-row ${active}" data-index="${index}">
              <span class="lm-color-dot" style="background:${colorForShape(shape)}"></span>
              <button type="button" class="text-left min-w-0" data-select-index="${index}">
                <span class="det-name">${escapeHtml(label)}</span>
                <span class="det-meta">
                  ${confChip}
                  <span class="det-chip">${sourceText(shape.source)}</span>
                  <span class="det-chip">${decisionText(shape.decision)}</span>
                  <span class="det-chip">${shapeName(shape.shape_type)}</span>
                </span>
                ${conf === null ? "" : `<span class="det-progress"><span style="width:${confWidth}%"></span></span>`}
              </button>
              <span class="det-chip">${index + 1}</span>
              <div class="det-label-editor">
                <input class="det-label-input" data-label-input="${index}" value="${escapeHtml(shape.label || "")}" placeholder="\u8f93\u5165\u6807\u7b7e\u540d">
                <div class="det-label-suggestions" data-label-suggestions="${index}"></div>
              </div>
              <div class="det-card-actions">
                <button type="button" class="det-mini-btn accept" data-action="accepted" data-index="${index}">\u63a5\u53d7</button>
                <button type="button" class="det-mini-btn modify" data-action="modified" data-index="${index}">\u4fee\u6539</button>
                <button type="button" class="det-mini-btn reject" data-action="rejected" data-index="${index}">\u62d2\u7edd</button>
              </div>
            </div>`;
        })
        .join("");
      [...objectList.querySelectorAll("[data-select-index]")].forEach((button) => {
        button.addEventListener("click", () => selectShape(shapes[Number(button.dataset.selectIndex)]));
      });
      [...objectList.querySelectorAll("[data-action]")].forEach((button) => {
        button.addEventListener("click", () => {
          const shape = shapes[Number(button.dataset.index)];
          selectShape(shape);
          setShapeDecision(shape, button.dataset.action);
          if (button.dataset.action === "modified") setMode("select");
        });
      });
      [...objectList.querySelectorAll("[data-label-input]")].forEach((input) => {
        input.addEventListener("click", (event) => event.stopPropagation());
        input.addEventListener("focus", () => renderInlineLabelSuggestions(input));
        input.addEventListener("input", () => renderInlineLabelSuggestions(input));
        input.addEventListener("blur", () => {
          window.setTimeout(() => hideInlineLabelSuggestions(input), 120);
        });
        input.addEventListener("keydown", (event) => {
          event.stopPropagation();
          if (event.key === "Enter") input.blur();
        });
        input.addEventListener("change", () => {
          const shape = shapes[Number(input.dataset.labelInput)];
          selectShape(shape);
          renameShape(shape, input.value.trim());
        });
      });
    }
    if (labelSelect && activeShape) labelSelect.value = activeShape.label || "";
    renderLabelList();
  }

  // 加载标签：合并任务预置标签与接口返回的标签
  async function loadLabels() {
    if (labelSelect) labelSelect.innerHTML = `<option value="">${TXT.chooseLabel}</option>`;
    seedTaskLabels();
    try {
      const response = await fetch(`/api/tasks/${taskId}/meta_data/`);
      const data = await response.json();
      const apiLabels = Array.isArray(data.labels) ? data.labels : [];
      const taskLabels = Array.isArray(window.detectionTaskLabels) ? window.detectionTaskLabels : [];
      loadedLabels = [];
      [...taskLabels, ...apiLabels].forEach(rememberLabel);
      refreshLabelSelectOptions();
    } catch (error) {
      console.warn("load labels failed", error);
    }
  }

  // 判断图片校验状态文案（已校验/未校验）
  function statusLabel(item) {
    const raw = String(item.raw_status || item.status || "").toLowerCase();
    return raw.includes("verified") && !raw.includes("unverified") ? TXT.verified : TXT.unverified;
  }

  // 当前是否处于“已标注视图”模式
  function isLabeledView() {
    return (window.detectionViewMode || "labeled") === "labeled";
  }

  // 刷新样本信息区：模式徽标、不确定性、策略、模型版本、优先级
  function updateSampleInfo(item) {
    const uncertainty = Number(item?.uncertainty_score || 0);
    const active = uncertainty >= ACTIVE_UNCERTAINTY;
    if (modeBadge) {
      modeBadge.textContent = active ? TXT.activeLearning : TXT.precheck;
      modeBadge.classList.toggle("hot", active);
      modeBadge.classList.toggle("ok", !active);
    }
    if (acceptAllBtn) {
      acceptAllBtn.disabled = false;
      acceptAllBtn.title = "";
    }
    if (sampleUncertainty) sampleUncertainty.textContent = fmt(uncertainty, 4);
    if (sampleStrategy) sampleStrategy.textContent = active ? "Entropy / Margin" : "Pseudo Label";
    if (sampleModelVersion) sampleModelVersion.textContent = item?.model_version || item?.algorithm_version || "v1";
    if (samplePriority) samplePriority.textContent = priorityText(uncertainty);
  }

  // 刷新队列信息：当前进度、总数、优先级、预计剩余
  function updateQueueInfo() {
    const position = currentPage > 0 && currentItemIndex >= 0
      ? (currentPage - 1) * 12 + currentItemIndex + 1
      : 0;
    const total = currentTotalCount || currentItems.length || 0;
    const remaining = Math.max(0, total - position);
    const priority = priorityText(currentUncertainty());
    if (queueMeta) {
      queueMeta.textContent = `${position || "-"} / ${total || "-"} | \u961f\u5217\u4f18\u5148\u7ea7: ${priority} | \u9884\u8ba1\u5269\u4f59: ${remaining} \u5f20`;
    }
  }

  // 打开指定图片：重置状态、载入标注、设置图片源并在 onload 后渲染
  function openImage(item) {
    window.currentImageItem = item;
    currentItemIndex = currentItems.findIndex((candidate) => String(candidate.id) === String(item.id));
    undoStack = [];
    hasUnsavedChanges = false;
    updateUndoState();
    zoomScale = 1;
    imageEl.style.width = "";
    imageEl.style.height = "";
    imageEl.style.maxWidth = "";
    imageEl.style.maxHeight = "";
    baseImageSize = { width: 0, height: 0 };
    setAnnotationsVisible(true);
    clearOverlays();
    shapes = Array.isArray(item.annotations) ? item.annotations.map((annotation) => annotationToShape(annotation, item)) : [];
    collectLabelsFromShapes(shapes);
    clearPlaceholderShapeLabels();

    imageEl.src = item.relative_image_url || item.image_url || "";
    currentName.textContent = item.image_name || TXT.unnamed;
    currentMeta.textContent = `${TXT.shape}: ${shapes.length} / ${TXT.uncertainty}: ${fmt(item.uncertainty_score)}`;
    updateSampleInfo(item);
    updateQueueInfo();

    imageEl.onload = () => {
      imageEl.style.width = "";
      imageEl.style.height = "";
      imageEl.style.maxWidth = "";
      imageEl.style.maxHeight = "";
      baseImageSize = {
        width: imageEl.clientWidth,
        height: imageEl.clientHeight,
      };
      setZoom(1);
      shapes.forEach(renderShape);
      rerenderShapes();
      selectShape(null);
      updateShapePanel();
      centerCanvas();
    };

    [...imageList.querySelectorAll(".det-image-row")].forEach((row) => {
      row.classList.toggle("active", row.dataset.itemId === String(item.id));
    });
  }

  // 渲染左侧图片列表，并自动打开指定（或第一张）图片
  function renderImageList(items, preferredItemId = null, preferredIndex = 0) {
    currentItems = items;
    currentItemIndex = -1;
    if (!items.length) {
      imageList.innerHTML = `<div class="det-empty">${TXT.noImage}</div>`;
      clearOverlays();
      imageEl.removeAttribute("src");
      currentName.textContent = TXT.noImage;
      currentMeta.textContent = `${TXT.shape}: 0 / ${TXT.uncertainty}: -`;
      updateQueueInfo();
      return;
    }

    imageList.innerHTML = items
      .map((item) => {
        const url = item.relative_image_url || item.image_url || "";
        const uncertainty = Number(item.uncertainty_score || 0);
        const hotClass = uncertainty >= ACTIVE_UNCERTAINTY ? "hot" : "ok";
        const annotationCount = Array.isArray(item.annotations) ? item.annotations.length : 0;
        const name = escapeHtml(item.image_name || TXT.image);
        const uncertaintyChip = isLabeledView()
          ? ""
          : `<span class="det-chip ${hotClass}">${TXT.uncertainty} ${fmt(uncertainty)}</span>`;
        return `
          <button type="button" class="det-image-row" data-item-id="${item.id}">
            ${url ? `<img src="${escapeHtml(url)}" alt="${name}">` : '<div class="bg-gray-100 border"></div>'}
            <span class="min-w-0">
              <span class="det-name">${name}</span>
              <span class="det-meta">
                ${uncertaintyChip}
                <span class="det-chip">${annotationCount} ${TXT.shape}</span>
                <span class="det-chip">${statusLabel(item)}</span>
              </span>
            </span>
          </button>`;
      })
      .join("");

    [...imageList.querySelectorAll(".det-image-row")].forEach((row, index) => {
      row.addEventListener("click", () => navigateToImage(items[index]));
    });
    const preferredItem = preferredItemId
      ? items.find((item) => String(item.id) === String(preferredItemId))
      : null;
    const fallbackIndex = Math.max(0, Math.min(items.length - 1, Number(preferredIndex) || 0));
    openImage(preferredItem || items[fallbackIndex]);
  }

  // 渲染分页控件（上一页/下一页）并绑定翻页事件
  function renderPager(data) {
    const page = Number(data.page || 1);
    const totalPages = Number(data.total_pages || 1);
    currentTotalPages = totalPages;
    pager.innerHTML = `
      <button type="button" class="lm-button" id="det-prev-page" ${page <= 1 ? "disabled" : ""}>${TXT.pagePrev}</button>
      <span class="text-xs text-slate-600">${page} / ${totalPages}</span>
      <button type="button" class="lm-button" id="det-next-page" ${page >= totalPages ? "disabled" : ""}>${TXT.pageNext}</button>
    `;
    document.getElementById("det-prev-page")?.addEventListener("click", () => navigateToPage(page - 1));
    document.getElementById("det-next-page")?.addEventListener("click", () => navigateToPage(page + 1));
  }

  // 按视图模式/状态/排序构造查询参数，请求某页图片数据
  async function fetchDetectionPage(page = 1) {
    const params = new URLSearchParams({ page: String(page), page_size: "12" });
    const viewMode = window.detectionViewMode || "labeled";
    if (viewMode === "labeled") params.set("labeled_only", "1");
    if (viewMode === "manual" || viewMode === "unlabeled") {
      params.set("predicted_only", "1");
      params.set("status", "unverified");
    } else {
      const statusValue = statusFilter?.value || "all";
      if (statusValue !== "all") params.set("status", statusValue);
    }

    const sortValue = sortSelect?.value || "uncertainty:desc";
    if (sortValue) params.set("sort", sortValue);

    const response = await fetch(`/tasks/${taskId}/data/?${params.toString()}`);
    if (!response.ok) throw new Error(TXT.loadFail);
    return response.json();
  }

  // 加载指定页：请求数据后渲染列表与分页
  async function loadDetectionPage(page = 1, preferredItemId = null, preferredIndex = 0) {
    currentPage = Math.max(1, page);
    imageList.innerHTML = `<div class="det-empty">${TXT.loading}</div>`;
    try {
      const data = await fetchDetectionPage(currentPage);
      currentTotalCount = Number(data.total_count || 0);
      renderImageList(data.results || [], preferredItemId, preferredIndex);
      renderPager(data);
    } catch (error) {
      console.error(error);
      imageList.innerHTML = `<div class="det-empty text-red-600">${TXT.loadFail}</div>`;
    }
  }

  // 调用页面的全局统计刷新函数更新任务进度
  async function refreshTaskStats() {
    if (typeof window.refreshTaskStatistics !== "function") return;
    try {
      await window.refreshTaskStatistics(taskId);
    } catch (error) {
      console.warn("refresh task stats failed", error);
    }
  }

  // 保存标注：POST 到保存接口，成功后清空撤销栈与未保存标记
  async function persistShapes(modeOverride = null) {
    if (!window.currentImageItem) {
      window.showMessage(TXT.saveFirst, "warning");
      return null;
    }
    const response = await fetch(`/tasks/${taskId}/detection/save/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": window.getCookie("csrftoken"),
      },
      body: JSON.stringify({
        item_id: window.currentImageItem.id,
        mode: modeOverride || (isActiveLearningSample() ? "active_learning" : "pre_annotation_check"),
        annotations: shapes.map(shapeToPayload),
      }),
    });
    const data = await response.json();
    if (!response.ok || data.status !== "success") {
      throw new Error(data.message || TXT.saveFail);
    }
    undoStack = [];
    hasUnsavedChanges = false;
    updateUndoState();
    return data;
  }

  // 切换图片/翻页前的自动保存（有未保存改动时执行）
  async function saveBeforeLeaving() {
    if (!window.currentImageItem || !hasUnsavedChanges) return true;
    try {
      await persistShapes("auto_save");
      return true;
    } catch (error) {
      console.error(error);
      window.showMessage(error.message || TXT.saveFail, "error");
      return false;
    }
  }

  // 切换到另一张图片（先自动保存当前改动）
  async function navigateToImage(item) {
    if (!item || String(item.id) === String(window.currentImageItem?.id)) return;
    if (!(await saveBeforeLeaving())) return;
    openImage(item);
  }

  // 翻到指定页（先自动保存当前改动）
  async function navigateToPage(page) {
    if (!(await saveBeforeLeaving())) return;
    await loadDetectionPage(page);
  }

  // 点击“保存”按钮：保存后提示并刷新当前页（停留在原图片）
  async function saveShapes() {
    try {
      await persistShapes();
      window.showMessage(TXT.saved, "success");
      loadDetectionPage(currentPage, window.currentImageItem?.id);
    } catch (error) {
      console.error(error);
      window.showMessage(error.message || TXT.saveFail, "error");
    }
  }

  // 完成当前图片：接受所有非拒绝框、保存，然后跳到下一张/下一页
  async function completeCurrentImage() {
    if (!window.currentImageItem) {
      window.showMessage(TXT.saveFirst, "warning");
      return;
    }
    shapes.forEach((shape) => {
      if (shape.decision !== "rejected") shape.decision = "accepted";
    });
    const nextItem = currentItemIndex >= 0 ? currentItems[currentItemIndex + 1] : null;
    const nextIndex = currentItemIndex >= 0 ? currentItemIndex : 0;
    const nextPage = nextItem ? currentPage : Math.min(currentPage + 1, currentTotalPages);
    try {
      await persistShapes("manual_complete");
      await refreshTaskStats();
      window.showMessage(TXT.completed, "success");
      if (nextItem) {
        await loadDetectionPage(currentPage, nextItem.id, nextIndex);
      } else if (nextPage !== currentPage) {
        await loadDetectionPage(nextPage);
      } else {
        await loadDetectionPage(currentPage, null, nextIndex);
      }
    } catch (error) {
      console.error(error);
      window.showMessage(error.message || TXT.saveFail, "error");
    }
  }

  // 绘制多边形/线的实时预览（黄色虚线）
  function renderDraft(extraPoint = null) {
    removeDraft();
    if (!draftPoints.length && !extraPoint) return;
    sizeOverlay();
    const layer = ensureSvg();
    const normalizedExtra = extraPoint
      ? (Array.isArray(extraPoint) ? extraPoint : [extraPoint.x, extraPoint.y])
      : null;
    const points = normalizedExtra ? [...draftPoints, normalizedExtra] : [...draftPoints];
    if (!points.length) return;
    draftEl = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    draftEl.setAttribute("points", svgPoints(points));
    draftEl.setAttribute("fill", "none");
    draftEl.setAttribute("stroke", "#ffcc00");
    draftEl.setAttribute("stroke-width", "2");
    draftEl.setAttribute("stroke-dasharray", "5 4");
    draftEl.setAttribute("pointer-events", "none");
    layer.appendChild(draftEl);
  }

  // 新增一个形状（默认人工来源），压入撤销栈、渲染并选中
  function addShape(shape) {
    pushUndo();
    shapes.push({
      source: "manual",
      decision: "manual",
      ...shape,
    });
    renderShape(shapes[shapes.length - 1]);
    selectShape(shapes[shapes.length - 1]);
  }

  // 收尾多边形（至少 3 个点）
  function finishPolygon() {
    if (draftPoints.length < 3) return;
    addShape({
      shape_type: "polygon",
      label: labelSelect?.value || "",
      points: [...draftPoints],
    });
    draftPoints = [];
    removeDraft();
    updateFinishPolygonState();
    updateUndoState();
  }

  // 收尾直线（取前 2 个点）
  function finishLine() {
    if (draftPoints.length < 2) return;
    addShape({
      shape_type: "line",
      label: labelSelect?.value || "",
      points: [...draftPoints.slice(0, 2)],
    });
    draftPoints = [];
    removeDraft();
    updateFinishPolygonState();
    updateUndoState();
  }

  // 根据草稿点数决定“完成多边形”按钮是否可用
  function updateFinishPolygonState() {
    if (!finishPolygonBtn) return;
    finishPolygonBtn.disabled = !(activeTool === "polygon" && draftPoints.length >= 3);
  }

  // 两点间归一化距离（用于判断多边形是否闭合成环）
  function distance(a, b) {
    return Math.hypot(clamp01(a[0]) - clamp01(b[0]), clamp01(a[1]) - clamp01(b[1]));
  }

  // —— 事件绑定区 ——
  // 工具按钮：切换当前标注工具
  toolButtons.forEach((button) => {
    button.addEventListener("click", () => setTool(button.dataset.tool || "rectangle"));
  });
  finishPolygonBtn?.addEventListener("click", finishPolygon);
  // 右侧标签页切换（对象面板/标签面板等）
  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      tabButtons.forEach((tab) => tab.classList.toggle("active", tab === button));
      document.querySelectorAll(".det-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `panel-${button.dataset.panel}`);
      });
    });
  });
  // 工具栏按钮：选择模式/删除/保存/完成/撤销/全部接受
  selectBtn?.addEventListener("click", () => setMode("select"));
  deleteBtn?.addEventListener("click", removeActiveShape);
  saveBtn?.addEventListener("click", saveShapes);
  completeImageBtn?.addEventListener("click", completeCurrentImage);
  undoBtn?.addEventListener("click", undoLastStep);
  // “全部接受”：把所有模型框标为已接受（拒绝的除外）
  acceptAllBtn?.addEventListener("click", () => {
    pushUndo();
    shapes.forEach((shape) => {
      if (shape.source === "model" && shape.decision !== "rejected") shape.decision = "accepted";
      positionShape(shape);
    });
    updateShapePanel();
  });
  // 上一张：本页内切换，到页首则跳到上一页最后一张
  prevImageBtn?.addEventListener("click", async () => {
    if (currentItemIndex > 0) {
      navigateToImage(currentItems[currentItemIndex - 1]);
      return;
    }
    if (currentItemIndex === 0 && currentPage > 1) {
      if (!(await saveBeforeLeaving())) return;
      await loadDetectionPage(currentPage - 1, null, 11);
    }
  });
  // 下一张：本页内切换，到页尾则跳到下一页第一张
  nextImageBtn?.addEventListener("click", async () => {
    if (currentItemIndex >= 0 && currentItemIndex < currentItems.length - 1) {
      navigateToImage(currentItems[currentItemIndex + 1]);
      return;
    }
    if (currentItemIndex >= 0 && currentPage < currentTotalPages) {
      if (!(await saveBeforeLeaving())) return;
      await loadDetectionPage(currentPage + 1, null, 0);
    }
  });
  // 缩放/适配窗口/显示隐藏标注框；筛选与排序变化时回到第 1 页
  zoomInBtn?.addEventListener("click", () => setZoom(zoomScale * 1.15));
  zoomOutBtn?.addEventListener("click", () => setZoom(zoomScale / 1.15));
  fitWindowBtn?.addEventListener("click", () => setZoom(1));
  toggleAnnotationsBtn?.addEventListener("click", () => setAnnotationsVisible(!annotationsVisible));
  statusFilter?.addEventListener("change", () => navigateToPage(1));
  sortSelect?.addEventListener("change", () => navigateToPage(1));
  // 窗口尺寸变化时重绘标注并刷新滚动状态
  window.addEventListener("resize", rerenderShapes);
  window.addEventListener("resize", updateCanvasOverflow);

  // 鼠标按下：矩形工具下开始拉框绘制
  container?.addEventListener("mousedown", (event) => {
    if (mode !== "draw" || activeTool !== "rectangle" || !window.currentImageItem) return;
    drawing = true;
    dragStart = toRelativeCoords(event.clientX, event.clientY);
  });

  // 鼠标移动：拖拽调整矩形手柄 / 预览多边形与线 / 拖拽绘制新矩形
  container?.addEventListener("mousemove", (event) => {
    if (rectEdit) {
      // 拖拽矩形四角手柄：根据手柄方位更新对应边并规范化坐标
      const current = toRelativeCoords(event.clientX, event.clientY);
      const { shape, handle, bounds } = rectEdit;
      let minX = bounds.minX;
      let minY = bounds.minY;
      let maxX = bounds.maxX;
      let maxY = bounds.maxY;
      if (handle.includes("n")) minY = current.y;
      if (handle.includes("s")) maxY = current.y;
      if (handle.includes("w")) minX = current.x;
      if (handle.includes("e")) maxX = current.x;
      const nextMinX = Math.min(minX, maxX);
      const nextMinY = Math.min(minY, maxY);
      const nextMaxX = Math.max(minX, maxX);
      const nextMaxY = Math.max(minY, maxY);
      shape.points = [[nextMinX, nextMinY], [nextMaxX, nextMaxY]];
      markShapeModified(shape);
      positionShape(shape);
      updateShapePanel();
      return;
    }
    if (mode === "draw" && ["polygon", "line"].includes(activeTool) && draftPoints.length) {
      const current = toRelativeCoords(event.clientX, event.clientY);
      renderDraft([current.x, current.y]);
      return;
    }
    if (!drawing || !dragStart) return;
    // 实时更新临时矩形框（未提交）
    const current = toRelativeCoords(event.clientX, event.clientY);
    const x = Math.min(dragStart.x, current.x);
    const y = Math.min(dragStart.y, current.y);
    const w = Math.abs(current.x - dragStart.x);
    const h = Math.abs(current.y - dragStart.y);
    if (!container._tmp) {
      const tmp = document.createElement("div");
      tmp.className = "bbox bbox-draft bbox-source-manual";
      container._tmp = tmp;
      container.appendChild(tmp);
    }
    const size = imageSize();
    container._tmp.style.left = `${x * size.width}px`;
    container._tmp.style.top = `${y * size.height}px`;
    container._tmp.style.width = `${w * size.width}px`;
    container._tmp.style.height = `${h * size.height}px`;
  });

  // 鼠标松开：结束矩形拖拽并生成标注框（过小的框忽略）
  window.addEventListener("mouseup", (event) => {
    if (rectEdit) {
      rectEdit = null;
      return;
    }
    if (!drawing || !dragStart) return;
    drawing = false;
    const end = toRelativeCoords(event.clientX, event.clientY);
    const x = Math.min(dragStart.x, end.x);
    const y = Math.min(dragStart.y, end.y);
    const w = Math.abs(end.x - dragStart.x);
    const h = Math.abs(end.y - dragStart.y);
    dragStart = null;
    container._tmp?.remove();
    container._tmp = null;
    if (w < 0.005 || h < 0.005) return;
    addShape({
      shape_type: "rectangle",
      label: labelSelect?.value || "",
      points: [[x, y], [x + w, y + h]],
    });
  });

  // 单击画布：点工具直接落点，多边形/线工具追加顶点（多边形点回起点自动闭合）
  container?.addEventListener("click", (event) => {
    if (mode === "select" || !window.currentImageItem) return;
    if (activeTool === "rectangle") return;
    const point = toRelativeCoords(event.clientX, event.clientY);
    if (activeTool === "point") {
      addShape({ shape_type: "point", label: labelSelect?.value || "", points: [[point.x, point.y]] });
      return;
    }
    draftPoints.push([point.x, point.y]);
    if (activeTool === "polygon" && draftPoints.length > 3) {
      const last = draftPoints[draftPoints.length - 1];
      const first = draftPoints[0];
      if (distance(last, first) < 0.02) {
        draftPoints.pop();
        finishPolygon();
        return;
      }
    }
    renderDraft();
    updateFinishPolygonState();
    updateUndoState();
    if (activeTool === "line" && draftPoints.length >= 2) finishLine();
  });

  // 双击画布：完成当前多边形绘制
  container?.addEventListener("dblclick", (event) => {
    if (mode !== "draw" || activeTool !== "polygon") return;
    event.preventDefault();
    finishPolygon();
  });

  // 键盘快捷键：Enter 完成多边形、Esc 取消草稿、Delete 删除、A 接受、R 拒绝、数字键快速套用标签
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && activeTool === "polygon") finishPolygon();
    if (event.key === "Escape") {
      draftPoints = [];
      removeDraft();
      updateFinishPolygonState();
      updateUndoState();
    }
    if ((event.key === "Delete" || event.key === "Backspace") && activeShape) removeActiveShape();
    if ((event.key === "a" || event.key === "A") && activeShape) setShapeDecision(activeShape, "accepted");
    if ((event.key === "r" || event.key === "R") && activeShape) setShapeDecision(activeShape, "rejected");
    const labelIndex = Number(event.key) - 1;
    if (labelIndex >= 0 && labelIndex < loadedLabels.length && activeShape) {
      pushUndo();
      activeShape.label = loadedLabels[labelIndex];
      rememberLabel(activeShape.label);
      if (labelSelect) labelSelect.value = activeShape.label;
      setShapeDecision(activeShape, activeShape.source === "model" ? "modified" : activeShape.decision, false);
    }
  });

  // 标签下拉切换：为当前选中框改名，模型框会被标记为已修改
  labelSelect?.addEventListener("change", () => {
    if (!activeShape) {
      renderLabelList();
      return;
    }
    pushUndo();
    activeShape.label = labelSelect.value;
    rememberLabel(activeShape.label);
    if (activeShape.source === "model" && activeShape.decision !== "accepted") {
      activeShape.source = "manual";
      activeShape.decision = "modified";
    }
    positionShape(activeShape);
    updateShapePanel();
  });

  // 页面初始化：加载标签、默认矩形工具与排序，加载第 1 页图片
  document.addEventListener("DOMContentLoaded", async () => {
    await loadLabels();
    setTool("rectangle");
    if (sortSelect) sortSelect.value = "uncertainty:desc";
    loadDetectionPage(1);
  });
})();
