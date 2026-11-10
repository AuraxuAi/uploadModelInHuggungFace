import math
import logging

import torch
import torch.nn as nn

from transformer_project.config import TrainConfig

logger = logging.getLogger(__name__)


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to lr_min."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.lr_min
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr_min + cosine * (cfg.lr - cfg.lr_min)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to matrices (not norms/biases)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8)
    logger.info(
        "Optimizer: AdamW | decay params: %.2fM | no-decay params: %.2fM",
        sum(p.numel() for p in decay) / 1e6,
        sum(p.numel() for p in no_decay) / 1e6,
    )
    return opt
