# Qwen 多模态图像分类标注工具

## 项目结构
```
├── configs/
│   └── config.json     # 配置文件 可自定义路径
├── VL-master/              # 模型源代码
├── qwen.py                 # 主程序入口，执行推理任务
├── image.png               # 图片，可在config中自定义路径
└── README.md               # 使用说明
```
## config文件
```
{
  "image_path": "/home/smallb/models/Qwen/image.png",
  "task_description": "你是一个图像分类专家，只需要对输入图像做出分类判断。请在我提供的类别中选择一个最合适的标签，并估算其置信度（百分比）。输出格式必须为：标签名 空格 置信度（如“狗 98%”）。不要输出描述、关键词或其他内容。",
  "label_choices": [
    "桃花",
    "菊花",
    "梅花"
  ],
  "restrict_labels": true,
  "generation_params": {
    "do_sample": true,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 512
  }
}
```
### 字段解释：
|字段名|含义|
|------|------|
|image_path	|输入图像的绝对路径
|task_description	|模型理解任务的自然语言描述
|label_choices	|允许的标签列表（仅当 restrict_labels 为 true 时生效）
|restrict_labels	|是否限定标签输出必须来自 label_choices
|generation_params	|控制大模型生成风格

|字段名|含义|
|------|------|
|do_sample|是否进行采样（影响生成的随机性）
|temperature|采样温度，越大越随机
|top_p|控制采样多样性
|max_new_tokens|最大生成长度

## 如何运行
进入环境

``` conda activate qwen ```

模型推理

```python qwen.py --config configs/config.json```

