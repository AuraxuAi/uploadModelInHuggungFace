import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_project.config import ModelConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU: FFN(x) = (Swish(W_gate . x) * (W_up . x)) . W_down
    Used by PaLM, LLaMA, Mistral, Gemma. ~2/3 the params of a standard FFN
    at equivalent capacity, since the hidden dim is rounded down accordingly.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(2 * cfg.d_ff / 3)
        hidden = (hidden + 63) // 64 * 64  # round to multiple of 64 for GPU efficiency

        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class StandardFFN(nn.Module):
    """Standard GELU FFN, kept as a simpler alternative to SwiGLU."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
