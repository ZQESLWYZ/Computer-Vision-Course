# CIFAR10 GAN Implementation

这是一个基于PyTorch的CIFAR10数据集GAN实现。

## 项目结构

```
GAN/
├── config.py              # 配置文件
├── generator.py           # 生成器网络
├── discriminator.py       # 判别器网络
├── utils.py              # 工具函数
├── train.py              # 训练脚本
├── generate_samples.py   # 生成样本脚本
└── README.md             # 说明文档
```

## 依赖要求

```bash
torch
torchvision
matplotlib
tqdm
numpy
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练过程中会：
- 自动下载CIFAR10数据集
- 创建`checkpoints`和`samples`目录
- 定期保存生成的样本和模型检查点

### 2. 生成样本

```bash
# 使用最新检查点生成样本
python generate_samples.py --show_plot

# 指定检查点路径
python generate_samples.py --checkpoint checkpoints/gan_epoch_100.pth --num_samples 100

# 只保存不显示
python generate_samples.py --save_path my_samples.png
```

## 网络架构

### 生成器
- 输入: 100维噪声向量
- 输出: 32x32x3 RGB图像
- 使用转置卷积层进行上采样
- 使用BatchNorm和ReLU激活函数

### 判别器
- 输入: 32x32x3 RGB图像
- 输出: 真实性概率
- 使用卷积层进行下采样
- 使用LeakyReLU激活函数

## 训练参数

- 批次大小: 128
- 学习率: 0.0002
- 优化器: Adam (β1=0.5, β2=0.999)
- 训练轮数: 100
- 噪声维度: 100

## 输出文件

- `checkpoints/`: 保存模型检查点
- `samples/`: 保存训练过程中的生成样本
- `data/`: CIFAR10数据集缓存

## 注意事项

- 确保有足够的GPU内存（建议4GB以上）
- 训练时间较长，建议使用GPU
- 生成的图像质量会随着训练轮数增加而改善
