import torch
import torch.nn as nn


def _initialize_conv_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
                for i in range(num_kernels)
            ]
        )
        if init_weight:
            _initialize_conv_weights(self)

    def forward(self, x):
        return torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(-1)


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(num_kernels // 2):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[1, 2 * i + 3],
                    padding=[0, i + 1],
                )
            )
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[2 * i + 3, 1],
                    padding=[i + 1, 0],
                )
            )
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            _initialize_conv_weights(self)

    def forward(self, x):
        return torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(-1)
