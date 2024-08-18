import torch, math
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Any


class L2Norm(torch.nn.Module):
    def __init__(self, p=2, dim=-1, eps=1e-12):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    # @staticmethod
    def forward(self, x: torch.Tensor):
        return torch.nn.functional.normalize(x, p=self.p, dim=self.dim, eps=self.eps)


class ElementwiseAdd(torch.nn.Module):
    """ Add module for a functional add"""
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        """
        Forward-pass routine for add op
        """
        return x + y


class FMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a: torch.Tensor, b: torch.Tensor):
        return torch.matmul(a, b)


class ElementwiseMul(torch.nn.Module):
    """ Multiply module for a functional multiply"""
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        """
        Forward-pass routine for multiply op
        """
        return x * y


class FCat(torch.nn.Module):
    """ Concat module for a functional concat"""
    def __init__(self, axis: int = 0):
        super(FCat, self).__init__()
        self._axis = axis

    # pylint:disable=arguments-differ
    def forward(self, *x) -> torch.Tensor:
        """
        Forward-pass routine for cat op
        """
        return torch.cat(x, dim=self._axis)


class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.mul = ElementwiseMul()
    
    def forward(self, x):
        return self.mul(x * self.sigmoid(x))
