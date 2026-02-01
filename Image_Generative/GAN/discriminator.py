import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_size=32, channels=3, features=64):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        
        self.main = nn.Sequential(
            # 输入: 32x32x3 -> 16x16xfeatures
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 1x1
            nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        validity = self.main(img)
        return validity.view(-1, 1)

def test_discriminator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(16, 3, 32, 32).to(device)
    disc = Discriminator().to(device)
    output = disc(x)
    print(f"Discriminator output shape: {output.shape}")
    assert output.shape == (16, 1)
    print("Discriminator test passed!")

if __name__ == "__main__":
    test_discriminator()
