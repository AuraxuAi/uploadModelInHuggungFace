"""Fine-tuning data layout and replay mixing (uses only .tolist() so it is framework-agnostic)."""
import numpy as np
import pytest

from transformer_project.data.dataset import MixedDataset, SFTPairDataset, TextChunkDataset, encode_sft_pair
from transformer_project.data.tokenizer import TokenizerWrapper


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "c.txt"
    corpus.write_text(("<question> def return import print x y z <answer> <file_sep> <commit_before> <commit_message> <commit_after>\n") * 200)
    return TokenizerWrapper.train([str(corpus)], vocab_size=300, save_path=str(d / "t.json"))


def test_layout_is_one_document_not_two(tok):
    prompt_ids, response_ids = encode_sft_pair(tok, "<question>def x", "<answer>return y")
    full = prompt_ids + response_ids
    assert full[0] == tok.bos_id                       # exactly one BOS, at the very start
    assert full.count(tok.bos_id) == 1
    assert full[-1] == tok.eos_id                      # EOS only at the end, so the model learns where to stop
    assert full.count(tok.eos_id) == 1                 # in particular NOT between prompt and response


def test_loss_only_on_response(tok):
    prompt_ids, response_ids = encode_sft_pair(tok, "<question>def x", "<answer>return y")
    ds = SFTPairDataset([(prompt_ids, response_ids)], ctx=64, pad_id=tok.eos_id)
    x, y = ds[0]
    x, y = x.tolist(), y.tolist()
    n_p = len(prompt_ids)
    assert y[: n_p - 1] == [-100] * (n_p - 1)          # prompt positions: no loss
    assert y[n_p - 1: n_p - 1 + len(response_ids)] == response_ids   # the first response token is predicted from the last prompt token
    assert y[n_p - 1 + len(response_ids):] == [-100] * (64 - (n_p - 1 + len(response_ids)))  # padding: no loss
    assert tok.eos_id in y                             # the stop token IS trained on
    assert len(x) == len(y) == 64


def test_mixed_dataset_ratio_and_contents(tok):
    prompt_ids, response_ids = encode_sft_pair(tok, "<question>a", "<answer>b")
    sft = SFTPairDataset([(prompt_ids, response_ids)] * 30, ctx=16, pad_id=tok.eos_id)
    replay = TextChunkDataset(np.arange(1, 16 * 100 + 2, dtype=np.uint16), ctx=16)   # 100 chunks, no -100 anywhere
    mixed = MixedDataset(sft, replay, replay_fraction=0.25, seed=1)
    assert len(mixed) == 40                            # 30 pairs + 10 replay = 25% replay
    n_replay = 0
    for i in range(len(mixed)):
        _, y = mixed[i]
        if -100 not in y.tolist():                     # replay chunks have a loss on every token
            n_replay += 1
    assert n_replay == 10
    assert len({int(i) for i in mixed.replay_idx}) == 10   # distinct chunks


def test_mixed_dataset_validates_fraction(tok):
    sft = SFTPairDataset([([1], [2])], ctx=8, pad_id=0)
    replay = TextChunkDataset(np.arange(100, dtype=np.uint16), ctx=8)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            MixedDataset(sft, replay, bad)
