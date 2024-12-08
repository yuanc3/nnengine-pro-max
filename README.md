# nnengine-pro-max

BUAA 2023 硕士高等软工课程设计
Based on: https://github.com/WLFJ/nnengine

## 项目结构

```
.
├── src
│   ├── core                     张量、模型等核心接口
│   │   ├── init.py              初始化函数
│   │   ├── nn.py                模型接口
│   │   └── tensor.py            张量接口
│   ├── data                     数据
│   │   ├── dataloader.py 
│   │   └── dataLoaders
│   │       ├── cifar10_dataloader.py           
│   │       ├── cifar100_dataloader.py
│   │       ├── imagenet_dataloader.py                
│   │       └── mnist_dataloader.py 
│   ├── models                   内置模型
│   │   ├── GoogLeNet.py         GoogLeNet
│   │   ├── ResNet.py            ResNet
│   │   ├── VGG.py               VGGNet
│   │   └── ViT.py               Vision Transformer
│   └── utils                    训练、测试工具
│       ├── evaluator.py         评估器
│       ├── optimizer.py         优化器
│       ├── scheduler.py         学习率管理器
│       └── trainer.py           训练器
├── train.py                     训练demo
├── run.py                       测试demo
└── res                          数据集存储位置
    ├── mnist                    
    └── ...
```

## 测试项目运行
可通过命令行运行run.py测试已训练好的模型,可使用device参数指定计算模式，其中resnet50的参数文件过大，可通过下面的网盘链接下载并存放至‘/tmp’文件夹下。

链接：https://pan.baidu.com/s/1v5jgoWO-gEXnSmWmuBwKdQ 
提取码：pr4c
```
usage: run.py [-h] [--dataset {mnist,cifar10,cifar100}] [--model {resnet18,resnet50,vit}] [--checkpoint CHECKPOINT] [--device DEVICE]

options:
  -h, --help                             show this help message and exit
  --dataset {mnist,cifar10,cifar100}     load data from dataset
  --model {resnet18,resnet50,vit}        chose a model
  --checkpoint CHECKPOINT                checkpoint dir
  --device DEVICE                        device mode
```
部分例子如下：
```
python run.py --dataset cifar100 --model resnet18
python run.py --dataset mnist --model resnet50
python run.py --dataset cifar10 --model vit
python run.py --dataset mnist --model vit
```
如果需要测试模型训练可运行train.py，其中预置了三种数据集的训练方案，可通过调整data参数切换数据集。