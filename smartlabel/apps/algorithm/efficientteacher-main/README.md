# EfficientTeacher 目标检测模块

本目录是 SmartLabel 平台调用 EfficientTeacher/YOLO 目标检测所需的最小运行子集，不再保留上游仓库的示例、部署、转换和文档文件。

## 平台入口

- `train.py`：训练入口，由 `smartlabel/apps/data_management/tasks.py` 通过 `conda run -n efficientteacher` 调用。
- `detect.py`：推理入口，训练完成后用于生成 YOLO txt 格式候选框。
- `val.py`：训练流程内部验证入口。

## 保留目录

- `configs/`：EfficientTeacher 配置定义和默认配置。
- `models/`：YOLO/EfficientTeacher 模型结构、损失和优化器。
- `trainer/`：监督训练和半监督训练循环。
- `utils/`：数据集、NMS、日志、绘图、设备选择等运行工具。

## 平台约定

- 任务配置由 SmartLabel 动态写入 `media/results/<task_id>/config.yaml`。
- 训练图片列表写入 `media/results/<task_id>/data_lists/`。
- 预训练权重默认读取 `models/efficient-yolov5l-obj365.pt`，也可以通过环境变量 `SMARTLABEL_EFFICIENTTEACHER_WEIGHTS` 指定。
- 输出权重位于 `media/results/<task_id>/task_<task_id>_effteacher*/weights/`。

## 已删除内容

以下内容不参与平台运行，已从本目录移除：

- `assets/`
- `data/`
- `deploy/`
- `scripts/`
- `export.py`
- `setup.py`
- 上游长 README/NOTICE
- Python 缓存目录

