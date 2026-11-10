import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    RoPE: encodes position by rotating Q and K vectors. No learnable
    parameters; extrapolates reasonably beyond the training context length.

    Note on the original notebook: it overrode `self.register_buffer` with a
    lambda that just did `setattr`, which meant the cos/sin caches were NOT
    real PyTorch buffers. That silently breaks `state_dict()` (checkpoints
    won't include them — harmless here since they're deterministic, but it's
    the kind of pattern that bites you the moment you add a real learned
    buffer next to it) and `.to(device)`/`.half()` calls won't move them
    automatically. This version registers real buffers.
    """

    def __init__(self, dim: int, max_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_len).float()
        freqs = torch.outer(pos, theta)              # (max_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)       # (max_len, dim)

        # persistent=False: deterministic from (dim, max_len, base), no need
        # to bloat checkpoints with it, but it DOES move with .to()/.cuda().
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int, offset: int = 0):
        """
        offset: starting position of this chunk (needed for correct RoPE
        angles during KV-cached generation, where q/k only contain the new
        token(s), not the full sequence from position 0).
        """
        if seq_len + offset > self.max_len:
            raise ValueError(
                f"Position {seq_len + offset} exceeds RoPE cache length {self.max_len}. "
                f"Increase max_len when constructing RotaryEmbedding."
            )
        cos = self.cos_cached[:, :, offset: offset + seq_len, :]
        sin = self.sin_cached[:, :, offset: offset + seq_len, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
