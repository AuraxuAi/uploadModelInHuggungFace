from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from transformer_project.data.tokenizer import TokenizerWrapper
from transformer_project.model.transformer import Transformer



def _encode_prompt(tokenizer, prompt: str) -> list:
    """Encode a prompt for continuation.

    The tokenizer's post-processor wraps every text as [BOS] ... [EOS], which is
    right for training documents but wrong for a prompt: a trailing [EOS] tells
    the model "this document is over", so it starts a new document instead of
    continuing the prompt. Keep [BOS], drop the final [EOS].
    """
    ids = tokenizer.encode(prompt)
    if len(ids) > 1 and tokenizer.eos_id is not None and ids[-1] == tokenizer.eos_id:
        ids = ids[:-1]
    return ids

def _top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k).values[..., -1, None]
        logits = torch.where(logits < kth_val, torch.full_like(logits, float("-inf")), logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_mask] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    return logits


@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer: TokenizerWrapper,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    # 1.0 = off. The old default (1.1) divides a token's score by 1.1**count every time it repeats, which wrecks
    # code (indentation, brackets, `self`, names must repeat). Only use it for prose, and keep it small.
    repetition_penalty: float = 1.0,
    strategy: str = "sample",
    beam_width: int = 4,
    use_kv_cache: bool = True,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    stop_strings: Optional[list] = None,
) -> str:
    """Generation strategies: 'greedy' | 'sample' (temperature/top-k/top-p) | 'beam'.

    stop_strings: stop early once the GENERATED text contains any of these (checked every 8 tokens, so the
    returned text may run a few tokens past the stop string; callers cut it at the first occurrence).
    """
    model.eval()

    if strategy == "beam":
        return _beam_search(model, tokenizer, prompt, device, max_new_tokens, beam_width, use_amp, amp_dtype)
    if strategy not in {"greedy", "sample"}:
        raise ValueError(f"Unknown generation strategy: {strategy}")

    ids = _encode_prompt(tokenizer, prompt)
    if not ids:
        raise ValueError("Prompt encoded to zero tokens.")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    seen: dict[int, int] = {}
    past_kvs = None

    prompt_len = len(ids)
    for step in range(max_new_tokens):
        inp = idx[:, -1:] if (use_kv_cache and past_kvs is not None) else idx[:, -model.cfg.context_len:]

        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            logits, _, past_kvs = model(inp, past_kvs=past_kvs if use_kv_cache else None, use_cache=use_kv_cache)
        logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            for tok_id, cnt in seen.items():
                penalty = repetition_penalty ** cnt
                if logits[0, tok_id] > 0:
                    logits[0, tok_id] /= penalty
                else:
                    logits[0, tok_id] *= penalty

        if strategy == "greedy":
            next_tok = logits.argmax(dim=-1, keepdim=True)
        else:
            if temperature <= 0:
                raise ValueError("temperature must be > 0 for sampling; use strategy='greedy' for temperature=0.")
            logits = logits / temperature
            logits = _top_k_top_p_filter(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        tok_id = next_tok.item()
        seen[tok_id] = seen.get(tok_id, 0) + 1
        idx = torch.cat([idx, next_tok], dim=1)

        if tokenizer.eos_id is not None and tok_id == tokenizer.eos_id:
            break
        if stop_strings and step % 8 == 7:
            generated = tokenizer.decode(idx[0, prompt_len:].tolist())
            if any(stop in generated for stop in stop_strings):
                break

    return tokenizer.decode(idx[0].tolist())


@torch.no_grad()
def _beam_search(
    model: Transformer,
    tokenizer: TokenizerWrapper,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    beam_width: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> str:
    """Simple beam search without KV caching (recomputes the full prefix each
    step). Fine for short generations; for long ones, batch the beams through
    the KV-cache path instead."""
    ids = _encode_prompt(tokenizer, prompt)
    beams = [(ids, 0.0)]  # (token_ids, cumulative log-prob)
    finished = []

    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in beams:
            idx = torch.tensor([seq[-model.cfg.context_len:]], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                logits, _, _ = model(idx)
            log_probs = F.log_softmax(logits[:, -1, :].float(), dim=-1)
            top_lp, top_idx = torch.topk(log_probs[0], beam_width)
            for lp, tok_id in zip(top_lp.tolist(), top_idx.tolist()):
                candidates.append((seq + [tok_id], score + lp))

        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for seq, score in candidates:
            if tokenizer.eos_id is not None and seq[-1] == tokenizer.eos_id:
                finished.append((seq, score))
            else:
                beams.append((seq, score))
            if len(beams) == beam_width:
                break
        if not beams:
            break

    all_candidates = finished + beams
    best_seq, _ = max(all_candidates, key=lambda c: c[1]) if all_candidates else (ids, 0.0)
    return tokenizer.decode(best_seq)
