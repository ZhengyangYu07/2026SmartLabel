# fixmatch
## config.json文件
config.json文件配置了模型的具体参数
关键说明
1. "image_dir"：自定义图像的目录路径。
2. "label_csv"：自定义标签文件（CSV 格式，包括 filename,class）
3. "num_classes"：类别数由传入csv自动判断并添加到json文件里
4. "out"：输出结果路径，输出包含model_best.tar其他checkpoint和标签名到数字的映射label2id.json
5. "total_steps"：为了加快可以减小"total_steps"，但是准确率会下降
## 调用时先调用train.py
输入config.json文件的目录
config.json文件内配置有模型大小参数，训练参数，数据目录等 
调用示例：

    python train.py --config config.json

## 然后调用test.py
传入
1. config.json文件的目录
2. model_best.tar训练得到的最佳模型目录
3. example.jpg需要判断的图片
4. label_map 标签名到数字的映射label2id.json

输出    

1. 预测标签
2. 置信度

调用示例：
   
    python test.py --config config.json  --checkpoint ./result/model_best.tar --image ./example.jpg --label_map ./result/label2id.json
