## 公开数据集构建程序

Note: 以下代码片段未考虑路径依赖，请从 bstool 根目录运行

### 1. 计算所有图像的分数，排序后存入 csv 文件
```
python image_num_height_angle_1024_v2 --remote source 
```

### 2. 从 CSV 文件中挑选出 3000 张有效图像
```
python create_public_training.py
```

### 3. 将数据集转换成 COCO 格式
```
python bs2coco_public_training.py
```

### 4. 模型评估

这里仅仅只放 segmentation 的评估代码，完整的评估过程融入到 mmdetv2-bc 中