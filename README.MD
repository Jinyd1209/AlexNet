## 从零实现AlexNet

### 环境配置
使用以下命令来加载环境配置
```cmd
   pip3 install -r requirements.txt
```
### 数据预处理
由于数据集较简单，数据的预处理部分也放在了训练文件中，主要使用torchvison中的transforms模块，具体信息可查阅pytorch官网

### 训练模型
1. 运行**split_data.py**脚本将数据集分为训练集train和验证集val,下载后的数据集存储在"flower_data"文件夹下
2. 运行**train.py**脚本对模型进行训练，由于该模型使用GPU进行训练，故需要先安装cuda
3. 训练完后会生成对应的.pth权重文件，将权重文件保存在根目录下以备预测时调用

### 模型预测
数据集中包含了五种不同种类的花，可自行下载同种类的花的图片输入到网络进行预测，若一次只预测一张图片可将花的图片保存到文件根目录并运行
**prediction.py**脚本得到预测种类和置信概率

点击链接下载花分类数据集 [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

模型论文参考可点击 [ImageNet Classification with Deep Convolutional
Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)