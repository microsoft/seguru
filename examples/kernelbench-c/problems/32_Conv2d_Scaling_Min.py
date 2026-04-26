import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A model that performs a 2D convolution, scales the result, and takes the
    minimum over the channel dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = x * self.scale_factor
        x = torch.min(x, dim=1, keepdim=True)[0]
        return x


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
