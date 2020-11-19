# AutomatedTesting2020

* Author: @NIL-zhuang
* Name: 庄子元
* ID: 181830266

选题方向： AI测试数据扩增

## 算法释义

使用几何变换（位置移动，扭曲，旋转）、图像扭曲（正态扭曲方法）、噪声（高斯模糊，椒盐噪声）等方法，增强数据。然后在基于MNIST的DNN模型上训练，验证模型的鲁棒性能有所提升。

## 所用第三方库及版本

1. python 3.8.6
2. tensorflow 2.3.1
3. opencv 4.5.0-1

## 程序入口

Projects文件里MNIST和CIFAR文件夹分别是两个数据集的控制程序

## 程序结构

```text
.
├── Data
├── Demo
├── Models
│   ├── CIFAR100
│   └── MNIST
├── Project
│   ├── CIFAR100
│   ├── MINIST
├── README.md
└── Report
    └── Report.md
```

* Data下存放的是扩增后的数据集
* Demo存放录制的视频
* Models存放助教给的模型和自己生成的模型
* Poject下存放对应数据集的扩增代码和验证代码
* Report下存放报告

## reference

[1] Shorten, C., Khoshgoftaar, T.M. A survey on Image Data Augmentation for Deep Learning. J Big Data 6, 60 (2019). https://doi.org/10.1186/s40537-019-0197-0
[2] Zhong, Zhun & Zheng, Liang & Kang, Guoliang & Li, Shaozi & Yang, Yi. (2017). Random Erasing Data Augmentation. Proceedings of the AAAI Conference on Artificial Intelligence. 34. 10.1609/aaai.v34i07.7000.
