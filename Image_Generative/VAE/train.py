import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from models.vae import ConvVAE, vae_loss

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3
LATENT_DIM = 20

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 创建模型
model = ConvVAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 创建保存目录
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# 训练函数
def train():
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = vae_loss(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item() / len(data):.6f}')
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> 轮次: {epoch} 平均损失: {avg_loss:.6f}')
    return avg_loss

# 测试函数
def test():
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            test_loss += vae_loss(recon_batch, data, mu, log_var).item()
            
            if i == 0:
                # 保存重构图像
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch[:n]])
                save_image(comparison.cpu(),
                         f'results/reconstruction_epoch_{epoch}.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f'====> 测试集损失: {test_loss:.6f}')
    return test_loss

# 生成样本函数
def generate_samples():
    model.eval()
    with torch.no_grad():
        # 从潜在空间采样
        sample = torch.randn(64, LATENT_DIM).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   f'results/sample_epoch_{epoch}.png')

# 保存图像的辅助函数
def save_image(tensor, filename, nrow=8):
    import torchvision.utils as vutils
    vutils.save_image(tensor, filename, nrow=nrow)

# 训练循环
train_losses = []
test_losses = []

print("开始训练VAE模型...")
for epoch in range(1, EPOCHS + 1):
    train_loss = train()
    test_loss = test()
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    # 每10轮保存一次模型和生成样本
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'checkpoints/vae_epoch_{epoch}.pth')
        generate_samples()
        print(f"模型已保存: checkpoints/vae_epoch_{epoch}.pth")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='训练损失')
plt.plot(test_losses, label='测试损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.title('VAE训练损失曲线')
plt.legend()
plt.grid(True)
plt.savefig('results/loss_curve.png')
plt.show()

# 保存最终模型
torch.save(model.state_dict(), 'checkpoints/vae_final.pth')
print("训练完成！最终模型已保存: checkpoints/vae_final.pth")
