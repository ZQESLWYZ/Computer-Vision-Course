import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=32, channels=3, features=64):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # 输入: latent_dim -> 4x4xfeatures*8
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            
            # 32x32 -> 32x32
            nn.ConvTranspose2d(features, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        img = self.main(z)
        return img

def test_generator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = torch.randn(16, 100).to(device)
    gen = Generator().to(device)
    output = gen(z)
    print(f"Generator output shape: {output.shape}")
    assert output.shape == (16, 3, 32, 32)
    print("Generator test passed!")

if __name__ == "__main__":
    test_generator()
