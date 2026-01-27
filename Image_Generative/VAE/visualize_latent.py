import torch
import matplotlib.pyplot as plt
import numpy as np
from models.vae import ConvVAE
import torchvision.utils as vutils
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载训练好的模型
def load_model(model_path='checkpoints/vae_final.pth', latent_dim=20):
    model = ConvVAE(latent_dim=latent_dim).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已加载: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return None
    model.eval()
    return model

# 生成潜在空间网格采样
def generate_latent_grid(model, n_rows=10, n_cols=10, latent_dim=20):
    """生成潜在空间的网格可视化"""
    model.eval()
    
    # 创建潜在空间的网格点
    # 只使用前两个维度进行可视化
    z1 = np.linspace(-3, 3, n_cols)
    z2 = np.linspace(-3, 3, n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    fig.suptitle('潜在空间网格采样 (前两个维度)', fontsize=16)
    
    with torch.no_grad():
        for i, z2_val in enumerate(z2):
            for j, z1_val in enumerate(z1):
                # 创建潜在向量，前两个维度为网格值，其余为0
                z = torch.zeros(1, latent_dim).to(device)
                z[0, 0] = z1_val
                z[0, 1] = z2_val
                
                # 生成图像
                generated = model.decode(z)
                generated = generated.cpu().squeeze().numpy()
                
                # 显示图像
                axes[i, j].imshow(generated, cmap='gray', vmin=0, vmax=1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].set_title(f'({z1_val:.1f},{z2_val:.1f})', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('results/latent_grid.png', dpi=150, bbox_inches='tight')
        plt.show()

# 随机采样大量图像
def generate_random_samples(model, n_samples=100, latent_dim=20):
    """生成大量随机样本"""
    model.eval()
    
    with torch.no_grad():
        # 生成随机潜在向量
        z = torch.randn(n_samples, latent_dim).to(device)
        
        # 生成图像
        generated = model.decode(z)
        
        # 保存为网格图像
        grid_size = int(np.sqrt(n_samples))
        if grid_size * grid_size < n_samples:
            grid_size += 1
        
        # 取前grid_size*grid_size个样本
        if generated.size(0) < grid_size * grid_size:
            # 补充样本
            additional_samples = grid_size * grid_size - generated.size(0)
            z_additional = torch.randn(additional_samples, latent_dim).to(device)
            generated_additional = model.decode(z_additional)
            generated = torch.cat([generated, generated_additional], dim=0)
        
        generated = generated[:grid_size * grid_size]
        
        # 保存网格图像
        vutils.save_image(generated, 'results/random_samples_grid.png', 
                         nrow=grid_size, normalize=False)
        
        # 显示
        plt.figure(figsize=(12, 12))
        grid = vutils.make_grid(generated, nrow=grid_size, normalize=False)
        plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
        plt.title(f'随机采样 {grid_size * grid_size} 个图像')
        plt.axis('off')
        plt.savefig('results/random_samples_display.png', dpi=150, bbox_inches='tight')
        plt.show()

# 潜在空间插值
def latent_interpolation(model, n_steps=20, latent_dim=20):
    """在潜在空间中进行插值"""
    model.eval()
    
    with torch.no_grad():
        # 生成两个随机潜在向量
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)
        
        # 在两个向量之间进行线性插值
        interpolations = []
        for alpha in np.linspace(0, 1, n_steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            generated = model.decode(z_interp)
            interpolations.append(generated.cpu())
        
        # 拼接所有插值图像
        interpolations = torch.cat(interpolations, dim=0)
        
        # 保存插值序列
        vutils.save_image(interpolations, 'results/latent_interpolation.png', 
                         nrow=n_steps, normalize=False)
        
        # 显示
        plt.figure(figsize=(20, 4))
        grid = vutils.make_grid(interpolations, nrow=n_steps, normalize=False)
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.title('潜在空间插值 (从左到右)')
        plt.axis('off')
        plt.savefig('results/latent_interpolation_display.png', dpi=150, bbox_inches='tight')
        plt.show()

# 2D潜在空间可视化（如果latent_dim=2）
def visualize_2d_latent(model, latent_dim=2):
    """专门为2D潜在空间创建可视化"""
    if latent_dim != 2:
        print("此功能仅适用于2D潜在空间")
        return
    
    model.eval()
    
    # 创建密集的2D网格
    n_points = 20
    x = np.linspace(-3, 3, n_points)
    y = np.linspace(-3, 3, n_points)
    
    fig, axes = plt.subplots(n_points, n_points, figsize=(20, 20))
    fig.suptitle('2D潜在空间完整可视化', fontsize=20)
    
    with torch.no_grad():
        for i, y_val in enumerate(y):
            for j, x_val in enumerate(x):
                z = torch.tensor([[x_val, y_val]], dtype=torch.float32).to(device)
                generated = model.decode(z)
                generated = generated.cpu().squeeze().numpy()
                
                axes[i, j].imshow(generated, cmap='gray', vmin=0, vmax=1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        
        plt.tight_layout()
        plt.savefig('results/latent_2d_full.png', dpi=200, bbox_inches='tight')
        plt.show()

# 主函数
def main():
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 模型参数
    latent_dim = 20
    
    # 加载模型
    model = load_model('checkpoints/vae_final.pth', latent_dim)
    if model is None:
        # 尝试加载最新的检查点
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.startswith('vae_epoch_') and f.endswith('.pth')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            model = load_model(f'checkpoints/{latest_checkpoint}', latent_dim)
        else:
            print("没有找到可用的模型文件")
            return
    
    print("开始潜在空间可视化...")
    
    # 1. 潜在空间网格采样（使用前两个维度）
    print("生成潜在空间网格采样...")
    generate_latent_grid(model, n_rows=10, n_cols=10, latent_dim=latent_dim)
    
    # 2. 大量随机采样
    print("生成大量随机样本...")
    generate_random_samples(model, n_samples=100, latent_dim=latent_dim)
    
    # 3. 潜在空间插值
    print("生成潜在空间插值...")
    latent_interpolation(model, n_steps=20, latent_dim=latent_dim)
    
    # 4. 如果是2D潜在空间，生成完整可视化
    if latent_dim == 2:
        print("生成2D潜在空间完整可视化...")
        visualize_2d_latent(model, latent_dim)
    
    print("所有可视化已完成！图像保存在 results/ 文件夹中。")

if __name__ == "__main__":
    main()
