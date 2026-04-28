# MixMatch

## 1. 说明

采用2019年Google提出的MixMatch，有bug待修改

### 1.1 数据格式

上传数据的格式，除了class_x可以修改，其他不能修改

```go
data/
|--labeled/
|--   |--class_1/
|--   |--class_2/
......
|--   |--class_n/
|--unlabeled/
|--   ......
```

### 1.2 json文件

|    参数名    |        说明        |
| :----------: | :----------------: |
| dataset_path | 上传数据存放的位置 |
|  output_csv  | 模型结果存放的位置和名称 |

### 1.3 主脚本中参数

| 参数名 |        说明        |
| :----: | :----------------: |
| --json | json文件的存放位置 |

### 1.4 默认数据集

00-19  cat

20-39  dog

40-59  car

60-79  plane

80-99  human

## 2. 使用方法

命令行中输出以下代码：

```python
your_python.exe  MixMatch.py --json your_config.json
```
