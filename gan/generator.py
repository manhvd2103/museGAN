import torch
import numpy as np

class GeneraterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)


class Generator(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(self):
        super().__init__()
        self.transconv0 = torch.nn.Sequential(
            GeneraterBlock(128, 2048, (2, 1, 1), (2, 1, 1)),
            GeneraterBlock(2048, 1024, (2, 1, 1), (2, 1, 1)),
            GeneraterBlock(1024, 512, (1, 2, 1), (1, 2, 1)),
            GeneraterBlock(512, 256, (1, 2, 1), (1, 2, 1)),
            GeneraterBlock(256, 128, (1, 1, 2), (1, 1, 2)),
            GeneraterBlock(128, 64, (1, 1, 2), (1, 1, 2)),
            GeneraterBlock(64, 32, (1, 1, 3), (1, 1, 1))
        )
        self.transconv1 = torch.nn.ModuleList([
            GeneraterBlock(32, 16, (1, 4, 1), (1, 4, 1))
            for _ in range(4)
        ])
        self.transconv2 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.ConvTranspose3d(16, 1, (1, 1, 12), (1, 1, 12)),
                torch.nn.Tanh()
            )
            for _ in range(4)
        ])

    def forward(self, x):
        x = x.view(-1, 128, 1, 1, 1)
        x = self.transconv0(x)
        x = [transconv(x) for transconv in self.transconv1]
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv2)], 1)
        x = x.view(-1, 4, 4 * 16, 72)
        return x
    
# if __name__ == '__main__':
#     in_put = torch.randn(1, 128)
#     generator = Generator()
#     out = generator(in_put)