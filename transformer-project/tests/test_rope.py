import torch

from transformer_project.model.rope import RotaryEmbedding


def test_rope_is_registered_as_real_buffer():
    """Regression test for the notebook bug where register_buffer was
    monkeypatched to a plain setattr, so cos/sin caches never showed up in
    state_dict() or moved with .to()/.half() automatically."""
    rope = RotaryEmbedding(dim=8, max_len=32)
    buffer_names = dict(rope.named_buffers()).keys()
    assert "cos_cached" in buffer_names
    assert "sin_cached" in buffer_names


def test_rope_offset_changes_rotation():
    """Applying RoPE at offset=0 vs offset=5 to the same vector must give
    different results — this is exactly the case the KV-cache decode path
    depends on getting right."""
    rope = RotaryEmbedding(dim=8, max_len=32)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    q0, k0 = rope(q, k, seq_len=1, offset=0)
    q5, k5 = rope(q, k, seq_len=1, offset=5)

    assert not torch.allclose(q0, q5)
    assert not torch.allclose(k0, k5)


def test_rope_out_of_range_offset_raises():
    rope = RotaryEmbedding(dim=8, max_len=16)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    raised = False
    try:
        rope(q, k, seq_len=1, offset=20)
    except ValueError:
        raised = True
    assert raised
