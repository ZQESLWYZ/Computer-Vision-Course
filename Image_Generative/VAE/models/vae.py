import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            # 输入: 1x28x28
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128x7x7
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256x7x7
            nn.ReLU(),
        )
        
        # 均值和方差
        self.fc_mu = nn.Linear(256 * 7 * 7, latent_dim)
        self.fc_var = nn.Linear(256 * 7 * 7, latent_dim)
        
        # 解码器
        self.decoder_input = nn.Linear(latent_dim, 256 * 7 * 7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # 128x7x7
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # 64x7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 1x28x28
            nn.Sigmoid()  # 输出到[0,1]范围
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def vae_loss(recon_x, x, mu, log_var):
    """
    VAE损失函数 = 重构损失 + KL散度
    """
    # 重构损失 (二元交叉熵)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL散度
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD
