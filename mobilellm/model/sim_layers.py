import torch, math
import torch.nn as nn
from .ops import ElementwiseMul
from typing import Optional, Tuple, List, Union, Any


class FRMSNorm(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)


class FROPE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.half_dim = config.head_dim // 2
        self.full_dim = config.head_dim

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: (nh, T, hs)
        # cos: (T, hs)
        # sin: (T, hs)
        # T = x.size(1)
        x1 = x[:, :, 0 : self.half_dim]                 # (nh, T, hs/2)
        x2 = x[:, :, self.half_dim : self.full_dim]     # (nh, T, hs/2)
        rotated = torch.cat((x2, x1), dim=-1)           # (nh, T, hs)
        roped = (x * cos) + (rotated * sin)
        return roped


class SLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, for_aimet=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.scale  = nn.Parameter(torch.ones(self.out_features)) 
        self.mul    = torch.mul if for_aimet else ElementwiseMul()

    def forward(self, x):
        return self.mul(self.scale, self.linear(x))
