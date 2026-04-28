
# 图像分类半监督学习系统说明文档


## 📁 项目结构

```
.
├── configs/                 配置文件目录
│   ├── config_img.json      图像分类配置示例
├── data/                    存放原始数据集
│   ├── custom               图像和部分标注数据
│   └── labels.csv           原始标签
├── embedding/              缓存的 clip 
├── model/                   模型定义目录
│   ├── CG3.py               图卷积模型（GC3）
│   ├── DiffMAP.py           扩散映射后的 MLP 分类器
│   └── LLGC.py              标签传播模型
├── outputs/                 输出预测结果目录
│   └── result.csv           预测结果
├── utils/                   工具模块
│   ├── encoder.py           clip 编码器封装
│   ├── graph_utils.py       图构建与 DiffMAP 生成
│   └── data_utils.py        数据加载与标签处理
└── train.py                 主训练与推理流程入口
```

---

##  运行方式

进入环境

``` conda activate img_classification ```

运行任务

```python train.py --config configs/config_img.json```



---
## 配置文件参数说明

| 参数名                   | 含义                       | 示例值                                                              |
| --------------------- | ------------------------ | ---------------------------------------------------------------- |
| `label_csv`           | 标签文件路径（CSV格式）            | `"/home/smallb/models/img_classfication/data/custom/labels.csv"` |
| `dataset_format`      | 数据格式           | `"csv"`                                                          |
| `image_dir`           | 图像数据所在的目录                | `"/home/smallb/models/img_classfication/data/custom/images"`     |
| `embedding_save_path` | 图像嵌入特征保存路径               | `"embedding/"`                                                   |
| `output_path`         | 最终分类或聚类结果输出路径            | `"output/"`                                                      |
| `batch_size`          | 图像特征提取时的批次大小             | `32`                                                             |
| `device`              | 使用设备（"cpu" 或 "cuda"）     | `"cuda"`                                                         |
| `cg3_k`               | CG3 图构建中 K 近邻数量          | `20`                                                             |
| `cg3_lr`              | CG3 模型学习率                | `1e-3`                                                           |
| `cg3_epochs`          | CG3 模型训练轮数               | `100`                                                            |
| `cg3_threshold`       | CG3 模型伪标签置信度阈值           | `0.8`                                                            |
| `cg3_hidden_dim`      | CG3 模型隐藏层维度              | `128`                                                            |
| `cg3_out_dim`         | CG3 模型输出特征维度             | `64`                                                             |
| `diffmap_k`           | DiffMAP 图构建中 K 近邻数量      | `20`                                                             |
| `diffmap_sigma`       | DiffMAP 高斯核参数            | `0.5`                                                            |
| `diffmap_components`  | DiffMAP 降维维度数            | `128`                                                            |
| `diffmap_epochs`      | DiffMAP 模型训练轮数           | `30`                                                             |
| `diffmap_lr`          | DiffMAP 学习率              | `0.01`                                                           |
| `llgc_alpha`          | LLGC 标签传播平衡系数（越接近 1 越平滑） | `0.5`                                                            |
| `manifold_lambda`     | Manifold 正则化强度参数         | `0.1`                                                            |
| `manifold_lr`         | Manifold 模型学习率           | `0.01`                                                           |
| `manifold_epochs`     | Manifold 模型训练轮数          | `30`                                                             |


---

## 输出结果说明

最终输出文件：
```
outputs/result.csv
```

包含以下字段：

| 列名             | 含义                                            |
| -------------- | --------------------------------------------- |
| `image`        | 未标注图像的文件名                                     |
| `pred_cg3`     |  CG3 的预测结果                           |
| `pred_diffmap` |  MLP 的预测结果                           |
| `pred_llgc`    |  LLGC 的预测结果                        |
| `pred_final`   | 三模型融合后 ManifoldClassifier 给出的最终预测结果      |
| `conf_final`   | ManifoldClassifier 输出中最终预测类别的 softmax 概率（置信度） |



