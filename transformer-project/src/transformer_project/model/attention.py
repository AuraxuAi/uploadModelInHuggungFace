from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_project.config import ModelConfig
from transformer_project.model.rope import RotaryEmbedding

KVCache = Tuple[torch.Tensor, torch.Tensor]


class MultiHeadAttention(nn.Module):
    """
    Grouped Query Attention (GQA): n_heads Q heads, n_kv_heads K/V heads.
      n_kv_heads == n_heads       -> standard MHA
      n_kv_heads == 1             -> Multi-Query Attention (MQA)
      1 < n_kv_heads < n_heads    -> GQA (LLaMA-2, Mistral)

    Bug fixed vs. the original notebook: during KV-cached generation, RoPE
    was applied to the newly-generated token using `cos_cached[:, :, :1, :]`
    — i.e. it always rotated as if the new token were at position 0,
    regardless of how many tokens had already been generated. That silently
    corrupts positional information for every token after the first one
    during cached generation. This version tracks `offset` = past sequence
    length and applies RoPE at the correct absolute position.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.attn_drop_p = cfg.attn_dropout
        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        # Cache generously beyond context_len so KV-cached generation can run
        # a bit past the training context without a hard crash; still bounded.
        self.rope = RotaryEmbedding(self.head_dim, max_len=max(cfg.context_len * 2, 2048), base=cfg.rope_base)


    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        b, n_kv, t, hd = x.shape
        return x.unsqueeze(2).expand(b, n_kv, self.n_rep, t, hd).reshape(b, n_kv * self.n_rep, t, hd)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)

        offset = past_kv[0].shape[2] if past_kv is not None else 0
        q, k = self.rope(q, k, seq_len=t, offset=offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        k_rep = self._repeat_kv(k)
        v_rep = self._repeat_kv(v)

        # Full prefill (no cache yet, more than one token): causal attention.
        # `is_causal=True` is exactly the lower-triangular mask for q_len == k_len,
        # but lets PyTorch use the fused FlashAttention kernel and avoids storing an
        # LxL mask per layer (an explicit attn_mask disables flash and, at an 8192
        # context, costs ~67 MB per layer). Single-token decode against an existing
        # cache: the new token is always the last position, so no mask is needed.
        is_causal = past_kv is None and t > 1

        out = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            attn_mask=None,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out), new_cache
