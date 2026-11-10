from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformer_project.config import ModelConfig
from transformer_project.model.attention import MultiHeadAttention, KVCache
from transformer_project.model.ffn import SwiGLUFFN
from transformer_project.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    Pre-norm gives more stable gradients in deep networks than post-norm.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = MultiHeadAttention(cfg)
        self.norm_ffn = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLUFFN(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        attn_out, cache = self.attn(self.norm_attn(x), past_kv=past_kv, use_cache=use_cache)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
        return x, cache
