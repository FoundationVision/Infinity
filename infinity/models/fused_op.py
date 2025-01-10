import gc
from copy import deepcopy
from typing import Union

import torch
from torch import nn as nn
from torch.nn import functional as F


#@torch.compile(fullgraph=True)
def fused_rms_norm(x: torch.Tensor, weight: nn.Parameter, eps: float):
    dtype = x.dtype
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps))) * weight
    return x.to(dtype=dtype)


#@torch.compile(fullgraph=True)
def fused_ada_layer_norm(C: int, eps: float, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    dtype = x.dtype
    x = F.layer_norm(input=x, normalized_shape=(C,), weight=None, bias=None, eps=eps)
    x = x.mul(scale.add(1)).add_(shift)
    return x.to(dtype=dtype)


#@torch.compile(fullgraph=True)
def fused_ada_rms_norm(C: int, eps: float, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    dtype = x.dtype
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps)))
    x = x.mul(scale.add(1)).add_(shift)
    return x.to(dtype=dtype)
