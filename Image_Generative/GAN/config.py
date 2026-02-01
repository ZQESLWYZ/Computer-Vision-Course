import torch

class Config:
    def __init__(self):
        # 数据参数
        self.batch_size = 128
        self.image_size = 32
        self.channels = 3
        self.latent_dim = 100
        
        # 训练参数
        self.epochs = 100
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.n_critic = 5
        
        # 模型参数
        self.generator_features = 64
        self.discriminator_features = 64
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 保存路径
        self.save_dir = 'checkpoints'
        self.sample_dir = 'samples'
        
        # 其他
        self.sample_interval = 10
        self.save_interval = 20
