import torch

from transformer_project.inference.generate import _top_k_top_p_filter


def test_top_k_filters_to_exactly_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtered = _top_k_top_p_filter(logits.clone(), top_k=2, top_p=1.0)
    finite_count = torch.isfinite(filtered).sum().item()
    assert finite_count == 2
    # The two largest values (5.0, 4.0) should survive.
    kept_values = filtered[torch.isfinite(filtered)].tolist()
    assert set(kept_values) == {5.0, 4.0}


def test_top_p_keeps_at_least_one_token():
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
    filtered = _top_k_top_p_filter(logits.clone(), top_k=0, top_p=0.01)
    assert torch.isfinite(filtered).sum().item() >= 1


def test_no_filtering_when_disabled():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    filtered = _top_k_top_p_filter(logits.clone(), top_k=0, top_p=1.0)
    assert torch.equal(filtered, logits)
