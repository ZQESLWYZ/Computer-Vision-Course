import torch
import argparse
import os
from config import Config
from generator import Generator
from utils import plot_generated_images, save_images

def load_generator(checkpoint_path, config):
    generator = Generator(
        latent_dim=config.latent_dim,
        img_size=config.image_size,
        channels=config.channels,
        features=config.generator_features
    ).to(config.device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"已加载生成器: {checkpoint_path}")
    else:
        print(f"未找到检查点文件: {checkpoint_path}")
        print("使用未训练的生成器")
    
    generator.eval()
    return generator

def main():
    parser = argparse.ArgumentParser(description='生成CIFAR10样本')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='生成器检查点路径')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='生成的样本数量')
    parser.add_argument('--save_path', type=str, default='generated_samples.png',
                        help='保存路径')
    parser.add_argument('--show_plot', action='store_true',
                        help='显示生成的图像')
    
    args = parser.parse_args()
    
    config = Config()
    
    # 如果没有指定检查点，尝试使用最新的
    if args.checkpoint is None:
        checkpoint_dir = config.save_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort()
                args.checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"使用最新检查点: {args.checkpoint}")
    
    # 加载生成器
    generator = load_generator(args.checkpoint, config) if args.checkpoint else None
    
    if generator is None:
        generator = Generator(
            latent_dim=config.latent_dim,
            img_size=config.image_size,
            channels=config.channels,
            features=config.generator_features
        ).to(config.device)
        generator.eval()
        print("使用未训练的生成器")
    
    # 生成样本
    with torch.no_grad():
        z = torch.randn(args.num_samples, config.latent_dim).to(config.device)
        generated_images = generator(z)
        
        # 保存图像
        save_images(generated_images, args.save_path)
        print(f"样本已保存到: {args.save_path}")
        
        # 显示图像
        if args.show_plot:
            plot_generated_images(generator, config.device, config.latent_dim, 
                                args.num_samples, args.save_path)

if __name__ == "__main__":
    main()
