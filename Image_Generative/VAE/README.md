# 基于全卷积和ReLU激活的VAE (MNIST)

本项目实现了一个基于全卷积神经网络和ReLU激活函数的变分自编码器(VAE)，用于MNIST手写数字数据集的训练和生成。

## 模型架构

### 编码器
- 4层卷积层，全部使用ReLU激活函数
- 卷积核大小：4x4 (前两层), 3x3 (后两层)
- 步长：2 (前两层), 1 (后两层)
- 输出通道：32 → 64 → 128 → 256
- 最终输出均值和对数方差

### 解码器
- 4层转置卷积层，全部使用ReLU激活函数（最后一层使用Sigmoid）
- 转置卷积核大小：3x3 (前两层), 4x4 (后两层)
- 步长：1 (前两层), 2 (后两层)
- 输出通道：256 → 128 → 64 → 32 → 1

### 损失函数
- 重构损失：二元交叉熵
- KL散度：正态先验的KL散度
- 总损失：重构损失 + KL散度

## 项目结构

```
VAE/
├── models/
│   └── vae.py          # VAE模型定义
├── train.py            # 训练脚本
├── requirements.txt    # 依赖包
├── README.md          # 项目说明
├── data/              # MNIST数据集（自动下载）
├── results/           # 训练结果图像
└── checkpoints/       # 模型检查点
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练模型

```bash
python train.py
```

## 训练参数

- **数据集**: MNIST
- **批次大小**: 128
- **训练轮数**: 100
- **学习率**: 1e-3
- **潜在空间维度**: 20
- **优化器**: Adam

## 输出文件

训练过程中会生成以下文件：

- `results/`: 包含重构图像、生成样本和损失曲线
- `checkpoints/`: 包含每10轮的模型检查点
- `data/`: MNIST数据集（自动下载）

## 模型特点

1. **全卷积架构**: 完全基于卷积操作，适合图像数据
2. **ReLU激活**: 所有隐藏层使用ReLU激活函数
3. **端到端训练**: 支持端到端的训练和推理
4. **可视化输出**: 自动保存重构图像和生成样本

## 使用示例

训练完成后，可以使用保存的模型进行图像生成：

```python
from models.vae import ConvVAE
import torch

# 加载模型
model = ConvVAE(latent_dim=20)
model.load_state_dict(torch.load('checkpoints/vae_final.pth'))
model.eval()

# 生成新图像
with torch.no_grad():
    z = torch.randn(1, 20)  # 随机潜在向量
    generated = model.decode(z)
```

## 性能指标

训练100轮后，模型能够：
- 重构MNIST数字图像
- 从潜在空间生成新的数字图像
- 保持潜在空间的连续性
