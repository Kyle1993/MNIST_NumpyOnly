# 使用说明  

没有使用现有的深度学习框架，只使用numpy，对mnist数据集进行预测  
模型将图片展开成向量，使用了三层全连接，手动计算梯度信息，并反向求导  

### 运行环境
python3.5  

### 包依赖
```python
import numpy
import pickle
import argparse
import six.moves
import gzip
import errno
import codecs
import PIL
import os,sys
```  

### 运行方法  

#### 路进设置
可以在file_path.py文件中进行路径设置，不设置将使用默认位置  
__mnist_path__: MNIST数据集（4个.gz文件）位置，默认下载至当前目录‘data'文件夹下  
__inference_path__: 需要预测图片所在文件夹位置，默认位置为当前目录‘inferenceData’文件夹下  

#### 模型训练
```python
python train.py -epochs <num of epochs> -batch_size <batch_size> -lr <lr>
```  

#### 模型检验
```python
python test.py
```  

#### 预测文件
```python
python inference.py
```  

### 文件说明
/data：MNIST数据集(4个.gz文件)  
/inference：待预测文件  
model.mod：模型参数，读取方法pickle  
training.log：训练日志  
test.log：测试日志  
Predict_result：预测结果  

### 运行结果
__参数：__ batch_size=128, lr=1e-4, epochs=10  
__训练集loss：__ 0.263623  
__测试集loss：__ 0.276538  
__测试集准确率：__ 0.9219  
