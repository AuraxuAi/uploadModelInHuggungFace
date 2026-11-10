import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_project.config import ModelConfig
from transformer_project.model.attention import KVCache
from transformer_project.model.block import TransformerBlock
from transformer_project.model.norm import RMSNorm

PastKVs = List[KVCache]


class Transformer(nn.Module):
    """Autoregressive decoder-only transformer: tok_emb -> N x block -> norm -> lm_head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2-style scaling of residual-branch output projections so that
        # variance doesn't blow up as depth increases.
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs: Optional[PastKVs] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[PastKVs]]:
        b, t = idx.shape
        if past_kvs is None and t > self.cfg.context_len:
            raise ValueError(
                f"Input length {t} exceeds context_len {self.cfg.context_len}. "
                f"Truncate the input or increase context_len in ModelConfig."
            )

        x = self.emb_drop(self.tok_emb(idx))

        new_kvs: Optional[PastKVs] = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, cache = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_kvs.append(cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss, new_kvs
