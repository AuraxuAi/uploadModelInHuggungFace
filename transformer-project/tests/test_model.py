import torch

from transformer_project.config import ModelConfig
from transformer_project.model.transformer import Transformer


def tiny_cfg(**overrides):
    base = dict(
        vocab_size=64, context_len=16, n_layers=2, n_heads=4,
        n_kv_heads=2, d_model=32, d_ff_mult=2.0, dropout=0.0, attn_dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_forward_shape():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss, _ = model(idx)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_computes_loss():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss, _ = model(idx, targets=targets)
    assert loss is not None
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_context_len_overflow_raises():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.context_len + 1))
    raised = False
    try:
        model(idx)
    except ValueError:
        raised = True
    assert raised


def test_weight_tying():
    cfg = tiny_cfg(tie_weights=True)
    model = Transformer(cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_kv_cache_matches_full_forward():
    """Generating token-by-token with a KV cache must produce the same
    logits as a single full forward pass over the same sequence. This is
    the invariant the original notebook's RoPE-offset bug silently broke."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = Transformer(cfg)
    model.eval()

    idx = torch.randint(0, cfg.vocab_size, (1, 6))

    with torch.no_grad():
        full_logits, _, _ = model(idx)

        past_kvs = None
        cached_logits = []
        for t in range(idx.shape[1]):
            step_input = idx[:, t: t + 1]
            logits, _, past_kvs = model(step_input, past_kvs=past_kvs, use_cache=True)
            cached_logits.append(logits[:, -1, :])
        cached_logits = torch.stack(cached_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-4, rtol=1e-3)


def test_num_parameters_positive():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    assert model.num_parameters() > 0
    assert model.num_parameters(non_embedding=True) < model.num_parameters()
