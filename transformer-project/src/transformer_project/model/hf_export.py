"""
Real HuggingFace Hub compatibility, as opposed to the original notebook's
"export_to_hf": that function saved a raw `state_dict` plus a hand-rolled
`config.json` with field names that don't match `transformers`' expectations
anywhere (`hidden_size`, `num_layers`, etc. are not read by any actual
`AutoModel` class) - so `AutoModel.from_pretrained(...)` on that export
would fail immediately. It called itself HF-compatible but wasn't.

This module defines a real `PretrainedConfig` and `PreTrainedModel`
subclass, so `save_pretrained` / `from_pretrained` behave the way anyone
familiar with `transformers` would expect, and the model can be pushed to
and loaded from the Hub with `trust_remote_code=True`.

Requires the optional `hf` extra: `pip install -e ".[hf]"`.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    from transformers import PretrainedConfig, PreTrainedModel
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "hf_export requires the 'transformers' package. Install with: pip install -e '.[hf]'"
    ) from e

from transformer_project.config import ModelConfig
from transformer_project.model.transformer import Transformer


class MiniTransformerConfig(PretrainedConfig):
    model_type = "mini_transformer"

    def __init__(
        self,
        vocab_size: int = 32000,
        context_len: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        d_model: int = 512,
        d_ff_mult: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.05,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        tie_weights: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_model = d_model
        self.d_ff_mult = d_ff_mult
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.tie_weights = tie_weights
        super().__init__(tie_word_embeddings=tie_weights, **kwargs)

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size, context_len=self.context_len, n_layers=self.n_layers,
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads, d_model=self.d_model,
            d_ff_mult=self.d_ff_mult, dropout=self.dropout, attn_dropout=self.attn_dropout,
            rope_base=self.rope_base, norm_eps=self.norm_eps, tie_weights=self.tie_weights,
        )

    @classmethod
    def from_model_config(cls, cfg: ModelConfig) -> "MiniTransformerConfig":
        return cls(
            vocab_size=cfg.vocab_size, context_len=cfg.context_len, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, d_model=cfg.d_model,
            d_ff_mult=cfg.d_ff_mult, dropout=cfg.dropout, attn_dropout=cfg.attn_dropout,
            rope_base=cfg.rope_base, norm_eps=cfg.norm_eps, tie_weights=cfg.tie_weights,
        )


class MiniTransformerForCausalLM(PreTrainedModel):
    config_class = MiniTransformerConfig

    def __init__(self, config: MiniTransformerConfig):
        super().__init__(config)
        self.transformer = Transformer(config.to_model_config())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        logits, loss, _ = self.transformer(input_ids, targets=labels)
        from transformers.modeling_outputs import CausalLMOutput

        return CausalLMOutput(loss=loss, logits=logits)

    @classmethod
    def from_local_checkpoint(
        cls, model_cfg: ModelConfig, state_dict: dict
    ) -> "MiniTransformerForCausalLM":
        """Build directly from an in-house ModelConfig + a Transformer state_dict
        (as produced by transformer_project.training.checkpoint), rather than
        going through the Hub round-trip."""
        hf_cfg = MiniTransformerConfig.from_model_config(model_cfg)
        hf_model = cls(hf_cfg)
        hf_model.transformer.load_state_dict(state_dict)
        return hf_model
