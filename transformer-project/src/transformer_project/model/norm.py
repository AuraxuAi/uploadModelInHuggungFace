import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Drops the mean-centering/bias term that LayerNorm has and only rescales
    by the root-mean-square of the activations, then applies a learned gain.
    Cheaper than LayerNorm and used by LLaMA, Mistral, Gemma, PaLM.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms
        return (x * self.weight).to(dtype)
