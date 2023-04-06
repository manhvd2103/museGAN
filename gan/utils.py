import torch

def initialize_weights(layer: torch.nn.Module, mean: float = 0.0, std: float = 0.02):
    if isinstance(layer, (torch.nn.Conv3d, torch.nn.ConvTranspose2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
    elif isinstance(layer, (torch.nn.Linear, torch.nn.BatchNorm2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
        torch.nn.init.constant_(layer.bias, 0)

