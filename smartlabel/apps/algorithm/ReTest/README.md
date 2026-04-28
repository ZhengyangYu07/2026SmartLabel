# 调用示例

```python main.py --config config/test.json```

# config说明

## dataset_config

描述数据集的信息

1. image_path：存放图片的地址，没什么用，之前的result.csv里面有具体地址

2. csv_path：读取result.csv的地址

3. label_column：label那列的名称

4. image_path_column：图像地址那列的名称

5. image_name_column：图像名称那列的名称

6. output_path：输出final_result.csv的地址

7. batch_size：加载数据的batch_size

## volminnet_config

volminnet的参数

1. architecture：选择用哪个网络架构作为backbone，目前只测试了resnet18

2. epoch

3. batch_size

4. lr

5. device

6. weight_decay

7. seed

## CWD_config

和volminnet差不多
