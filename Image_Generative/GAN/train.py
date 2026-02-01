import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from generator import Generator
from discriminator import Discriminator
from utils import get_cifar10_dataloader, save_images, create_directories, weights_init

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 创建目录
        create_directories(config.save_dir, config.sample_dir)
        
        # 初始化模型
        self.generator = Generator(
            latent_dim=config.latent_dim,
            img_size=config.image_size,
            channels=config.channels,
            features=config.generator_features
        ).to(self.device)
        
        self.discriminator = Discriminator(
            img_size=config.image_size,
            channels=config.channels,
            features=config.discriminator_features
        ).to(self.device)
        
        # 权重初始化
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        # 优化器
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=config.lr, 
            betas=(config.b1, config.b2)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=config.lr, 
            betas=(config.b1, config.b2)
        )
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 数据加载器
        self.dataloader = get_cifar10_dataloader(config.batch_size, config.image_size)
        
        # 损失历史记录
        self.g_losses = []
        self.d_losses = []
        
        print(f"训练设备: {self.device}")
        print(f"数据集大小: {len(self.dataloader.dataset)}")
        print(f"批次数: {len(self.dataloader)}")
    
    def train(self):
        print("开始训练GAN...")
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            self.generator.train()
            self.discriminator.train()
            
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
            
            for i, (imgs, _) in enumerate(pbar):
                batch_size = imgs.size(0)
                
                # 真实和虚假标签
                real = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)
                
                # 真实图像
                real_imgs = Variable(imgs.type(torch.FloatTensor)).to(self.device)
                
                # -----------------
                #  训练判别器
                # -----------------
                self.optimizer_D.zero_grad()
                
                # 真实图像损失
                real_loss = self.criterion(self.discriminator(real_imgs), real)
                
                # 生成虚假图像
                z = Variable(torch.randn(batch_size, self.config.latent_dim)).to(self.device)
                fake_imgs = self.generator(z)
                
                # 虚假图像损失
                fake_loss = self.criterion(self.discriminator(fake_imgs.detach()), fake)
                
                # 判别器总损失
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()
                
                # -----------------
                #  训练生成器
                # -----------------
                self.optimizer_G.zero_grad()
                
                # 生成虚假图像
                z = Variable(torch.randn(batch_size, self.config.latent_dim)).to(self.device)
                gen_imgs = self.generator(z)
                
                # 生成器损失（希望判别器将虚假图像判断为真实）
                g_loss = self.criterion(self.discriminator(gen_imgs), real)
                g_loss.backward()
                self.optimizer_G.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                
                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}'
                })
            
            # 计算平均损失
            avg_g_loss = epoch_g_loss / len(self.dataloader)
            avg_d_loss = epoch_d_loss / len(self.dataloader)
            
            # 记录损失
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch [{epoch+1}/{self.config.epochs}] - '
                  f'D_loss: {avg_d_loss:.4f} - G_loss: {avg_g_loss:.4f} - '
                  f'Time: {epoch_time:.2f}s')
            
            # 保存生成的样本
            self.save_samples(epoch + 1)
            
            # 每隔一定轮数绘制损失曲线
            if (epoch + 1) % self.config.sample_interval == 0:
                self.plot_losses(epoch + 1)
            
            # 保存模型
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_models(epoch + 1)
        
        print("训练完成!")
        # 最终绘制损失曲线
        self.plot_losses(self.config.epochs, final=True)
    
    def save_samples(self, epoch):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(64, self.config.latent_dim).to(self.device)
            generated_images = self.generator(z)
            
            # 创建epoch对应的子文件夹
            epoch_dir = os.path.join(self.config.sample_dir, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 保存图像网格
            save_path = os.path.join(epoch_dir, f'samples_epoch_{epoch}.png')
            save_images(generated_images, save_path)
            
            # 保存单独的图像（可选）
            for i in range(min(16, len(generated_images))):  # 保存前16张
                single_path = os.path.join(epoch_dir, f'sample_{i:03d}.png')
                single_img = generated_images[i:i+1]
                save_images(single_img, single_path, nrow=1)
            
            print(f"样本已保存: {epoch_dir}")
    
    def save_models(self, epoch):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }, os.path.join(self.config.save_dir, f'gan_epoch_{epoch}.pth'))
        print(f"模型已保存: epoch_{epoch}")
    
    def plot_losses(self, epoch, final=False):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.g_losses) + 1), self.g_losses, 'b-', label='Generator Loss', linewidth=2)
        plt.plot(range(1, len(self.d_losses) + 1), self.d_losses, 'r-', label='Discriminator Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'GAN Training Losses - Epoch {epoch}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 设置y轴范围，避免异常值影响可视化
        if len(self.g_losses) > 10:
            all_losses = self.g_losses + self.d_losses
            q1, q3 = np.percentile(all_losses, [25, 75])
            iqr = q3 - q1
            lower_bound = max(0, q1 - 1.5 * iqr)
            upper_bound = q3 + 1.5 * iqr
            plt.ylim(lower_bound, upper_bound)
        
        # 保存图像
        if final:
            save_path = os.path.join(self.config.sample_dir, 'final_loss_curve.png')
        else:
            save_path = os.path.join(self.config.sample_dir, f'loss_curve_epoch_{epoch}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"损失曲线已保存: {save_path}")
        
        # 实时显示（可选）
        if not final and epoch % 20 == 0:  # 每20个epoch显示一次
            plt.show()

if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()
