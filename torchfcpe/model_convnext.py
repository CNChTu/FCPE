from typing import Optional

import torch
from torch import nn


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        dilation (int, optional): Dilation factor for the depthwise convolution. Defaults to 1.
        kernel_size (int, optional): Kernel size for the depthwise convolution. Defaults to 7.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to 1e-6.
    """

    def __init__(
            self,
            dim: int,
            intermediate_dim: int,
            dilation: int = 1,
            kernel_size: int = 7,
            layer_scale_init_value: Optional[float] = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            dilation=dilation,
            padding=int(dilation * (kernel_size - 1) / 2),
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt layers

    Args:
        dim (int): Number of input channels.
        num_layers (int): Number of ConvNeXt layers.
        mlp_factor (int, optional): Factor for the intermediate layer dimensionality. Defaults to 4.
        dilation_cycle (int, optional): Cycle for the dilation factor. Defaults to 4.
        kernel_size (int, optional): Kernel size for the depthwise convolution. Defaults to 7.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to 1e-6.
    """

    def __init__(
            self,
            dim: int,
            num_layers: int = 20,
            mlp_factor: int = 4,
            dilation_cycle: int = 4,
            kernel_size: int = 7,
            layer_scale_init_value: Optional[float] = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.mlp_factor = mlp_factor
        self.dilation_cycle = dilation_cycle
        self.kernel_size = kernel_size
        self.layer_scale_init_value = layer_scale_init_value

        self.layers = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim,
                    dim * mlp_factor,
                    dilation=(2 ** (i % dilation_cycle)),
                    kernel_size=kernel_size,
                    layer_scale_init_value=1e-6,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
