"""Packing of fine-tuning examples: no example is split, labels follow the response mask."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_sft_packed as bsp  # noqa: E402

from transformer_project.data.dataset import PackedSFTDataset, encode_sft_pair  # noqa: E402
from transformer_project.data.tokenizer import TokenizerWrapper  # noqa: E402


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    (d / "c.txt").write_text(("<question> def return import print x y z <answer> <file_sep> value item count total\n") * 300)
    return TokenizerWrapper.train([str(d / "c.txt")], vocab_size=300, save_path=str(d / "t.json"))


def _pack(examples, row_len, n_bins=4):
    p, rows = bsp.Packer(row_len, pad_id=2, n_bins=n_bins), []
    for toks, mask in examples:
        rows += p.add(toks, mask)
    return rows + p.flush()


def test_packer_keeps_every_example_whole_and_loses_nothing():
    rng = random.Random(0)
    row_len = 512                                  # rows much longer than the examples, as in real use (2049 vs ~500)
    ex = []
    for i in range(400):
        n = rng.randint(20, 120)
        ex.append(([1] + [10 + i % 50] * (n - 2) + [2], [0, 0] + [1] * (n - 2)))
    rows = _pack(ex, row_len, n_bins=16)
    assert all(len(t) == row_len and len(m) == row_len for t, m, _ in rows)
    assert sum(r for _, _, r in rows) == sum(len(t) for t, _ in ex)          # every real token is kept
    # every row is a concatenation of whole examples: it starts with BOS (=1) and each EOS (=2) is followed by BOS or padding
    for t, m, real in rows:
        t = t.tolist()
        assert t[0] == 1
        for i in range(real - 1):
            if t[i] == 2:
                assert t[i + 1] == 1
    fill = sum(r for _, _, r in rows) / (len(rows) * row_len)
    assert fill > 0.95                                                        # best-fit keeps the waste low


def test_packed_dataset_labels(tok, tmp_path):
    pairs = [{"prompt": "<question> def x", "response": "return y"},
             {"prompt": "<question> print z", "response": "value item"},
             {"prompt": "<question> import", "response": "count total x y"}] * 20
    enc = [encode_sft_pair(tok, p["prompt"], p["response"]) for p in pairs]
    rows = _pack([(a + b, [0] * len(a) + [1] * len(b)) for a, b in enc], row_len=33)
    with open(tmp_path / "x.tokens.bin", "wb") as f:
        f.write(np.concatenate([t for t, _, _ in rows]).tobytes())
    with open(tmp_path / "x.mask.bin", "wb") as f:
        f.write(np.concatenate([m for _, m, _ in rows]).tobytes())
    ds = PackedSFTDataset(str(tmp_path / "x.tokens.bin"), str(tmp_path / "x.mask.bin"), ctx=32)
    assert len(ds) == len(rows)
    for i, (t, m, _) in enumerate(rows):
        x, y = ds[i]
        x, y = x.tolist(), y.tolist()
        assert len(x) == len(y) == 32
        assert x == t[:-1].tolist()
        for j in range(32):                      # label = next token only where the next token is a response token
            assert y[j] == (int(t[j + 1]) if m[j + 1] == 1 else -100)
    all_y = [v for i in range(len(ds)) for v in ds[i][1].tolist()]
    assert tok.eos_id in all_y                   # the stop token is trained on


def test_batched_encoding_matches_encode_sft_pair(tok):
    pairs = [{"prompt": "<question> def x", "response": "return y"}, {"prompt": "<file_sep> import", "response": "\n    value item\n"}]
    batched = bsp.encode_pairs(tok.tok, tok.bos_id, tok.eos_id, pairs)
    single = [encode_sft_pair(tok, p["prompt"], p["response"]) for p in pairs]
    assert batched == single


def test_dataset_rejects_misaligned_files(tmp_path):
    (tmp_path / "t.bin").write_bytes(np.zeros(10, dtype=np.uint16).tobytes())
    (tmp_path / "m.bin").write_bytes(np.zeros(10, dtype=np.uint8).tobytes())
    with pytest.raises(ValueError):
        PackedSFTDataset(str(tmp_path / "t.bin"), str(tmp_path / "m.bin"), ctx=8)   # 10 is not a multiple of 9
