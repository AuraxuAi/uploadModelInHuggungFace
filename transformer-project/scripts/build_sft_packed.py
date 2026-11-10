"""
Build a PACKED fine-tuning dataset of a few billion tokens (loss on responses only).

Why packing: at this size, padding every short example to the 2048-token window would waste
~70% of the GPU time. Here several examples share one row of (ctx+1) tokens; two parallel flat files
are written (read by PackedSFTDataset):

    <prefix>.train.tokens.bin   <prefix>.train.mask.bin     (mask 1 = response token = loss)
    <prefix>.val.tokens.bin     <prefix>.val.mask.bin
    <prefix>.meta.json          what went in (tokens per kind, fill rate, ...)

Layout of one example, identical to pretraining documents:  [BOS] prompt | response [EOS]

The data kinds and how they are produced are the same as scripts/build_sft_data.py:
  function       signature (+imports, docstring) -> body           (streamed from raw_python; fresh every time)
  stackoverflow  question -> answer                                (your stackoverflow.jsonl)
  commit         old file + message -> new file, SMALL edits only   (commitpack)
  contest        problem -> correct Python solution                 (code_contests + apps)

Token budgets are per kind (REAL tokens, i.e. without padding). Small sources run out: they are
repeated up to --max-repeat passes (reshuffled each pass); a kind that is exhausted simply ends early
and the shortfall is reported - it is never silently padded with other data.

Usage (CPU runtime, needs HF_TOKEN):
  python scripts/build_sft_packed.py --tokenizer artifacts/tokenizer_20b_32k.json \\
      --so-jsonl artifacts/stackoverflow.jsonl --out artifacts/sft_packed/sft2b
For offline tests any source can be a local jsonl: --local raw_python=file.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_corpus_20b as bc  # noqa: E402
import build_sft_data as sd  # noqa: E402


# ----------------------------------------------------------------------------
# packing
# ----------------------------------------------------------------------------
class Packer:
    """Best-fit packing into rows of `row_len` tokens, with a few rows open at once."""

    def __init__(self, row_len: int, pad_id: int, n_bins: int = 32):
        self.row_len, self.pad_id, self.n_bins = row_len, pad_id, n_bins
        self.bins: list[dict] = []

    def _close(self, b: dict):
        pad = self.row_len - len(b["t"])
        toks = np.array(b["t"] + [self.pad_id] * pad, dtype=np.uint16)
        mask = np.array(b["m"] + [0] * pad, dtype=np.uint8)
        return toks, mask, len(b["t"])

    def add(self, tokens: list[int], mask: list[int]):
        """Add one example; returns the rows completed by this call (usually 0 or 1)."""
        n, done = len(tokens), []
        fits = [b for b in self.bins if self.row_len - len(b["t"]) >= n]
        if fits:
            b = max(fits, key=lambda b: len(b["t"]))          # best fit = the fullest row that still has room
        else:
            if len(self.bins) >= self.n_bins:                  # no room anywhere: close the fullest row
                full = max(self.bins, key=lambda b: len(b["t"]))
                self.bins.remove(full)
                done.append(self._close(full))
            b = {"t": [], "m": []}
            self.bins.append(b)
        b["t"].extend(tokens)
        b["m"].extend(mask)
        return done

    def flush(self):
        out = [self._close(b) for b in self.bins if b["t"]]
        self.bins = []
        return out


def encode_pairs(tok, bos: int, eos: int, pairs: list[dict]):
    """Same layout as transformer_project.data.dataset.encode_sft_pair, but batched (multi-threaded)."""
    ps = tok.encode_batch([p["prompt"] for p in pairs], add_special_tokens=False)
    rs = tok.encode_batch([p["response"] for p in pairs], add_special_tokens=False)
    return [([bos] + a.ids, b.ids + [eos]) for a, b in zip(ps, rs)]


# ----------------------------------------------------------------------------
# pair sources
# ----------------------------------------------------------------------------
def function_stream(local, rng, seed, max_scan):
    src = bc.BY_NAME["raw_python"]
    seen, scanned = set(), 0
    for row in bc.iter_rows(src, local, seed=seed):
        scanned += 1
        if scanned > max_scan:
            return
        try:
            if src.keep is not None and not src.keep(row):
                continue
            pairs = sd.function_pairs(row["content"], rng, max_per_file=2)
        except KeyError:
            continue
        for p in pairs:
            h = hash((p["prompt"], p["response"]))
            if h not in seen:
                seen.add(h)
                yield p


def load_so(path, max_pairs):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = sd.so_pair(json.loads(line)["text"])
            if p:
                out.append(p)
                if len(out) >= max_pairs:
                    break
    return out


def load_commits(local, max_scan):
    src, out = bc.BY_NAME["commits"], []
    for i, row in enumerate(bc.iter_rows(src, local, seed=3)):
        if i >= max_scan:
            break
        p = sd.commit_pair(row)
        if p:
            out.append(p)
    return out


def load_contests(local, rng, max_scan):
    out = []
    for name, fn in (("code_contests", sd.contest_pairs), ("apps", sd.apps_pairs)):
        for i, row in enumerate(bc.iter_rows(bc.BY_NAME[name], local, seed=3)):
            if i >= max_scan:
                break
            out.extend(fn(row, rng, 4))
    return out


def cycle(pairs, rng, max_repeat):
    """Pass over `pairs` up to max_repeat times, reshuffled each pass."""
    pairs = list(pairs)
    for _ in range(max_repeat):
        rng.shuffle(pairs)
        yield from pairs


# ----------------------------------------------------------------------------
class Writer:
    def __init__(self, prefix: str, split: str, flush_rows: int = 2048):
        self.tf = open(f"{prefix}.{split}.tokens.bin", "wb")
        self.mf = open(f"{prefix}.{split}.mask.bin", "wb")
        self.buf_t, self.buf_m, self.rows, self.flush_rows = [], [], 0, flush_rows

    def add(self, toks, mask):
        self.buf_t.append(toks)
        self.buf_m.append(mask)
        self.rows += 1
        if len(self.buf_t) >= self.flush_rows:
            self._flush()

    def _flush(self):
        if self.buf_t:
            self.tf.write(np.concatenate(self.buf_t).tobytes())
            self.mf.write(np.concatenate(self.buf_m).tobytes())
            self.buf_t, self.buf_m = [], []

    def close(self):
        self._flush()
        self.tf.close()
        self.mf.close()


def human(n: float) -> str:
    return f"{n/1e9:,.2f}B" if n >= 1e9 else f"{n/1e6:,.1f}M" if n >= 1e6 else f"{n/1e3:,.1f}K"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default="artifacts/sft_packed/sft2b", help="output prefix")
    ap.add_argument("--so-jsonl", help="your stackoverflow.jsonl")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--tokens-functions", type=float, default=1.0e9)
    ap.add_argument("--tokens-so", type=float, default=0.45e9)
    ap.add_argument("--tokens-commits", type=float, default=0.15e9)
    ap.add_argument("--tokens-contests", type=float, default=0.06e9)
    ap.add_argument("--max-repeat", type=int, default=3, help="passes allowed over the small sources")
    ap.add_argument("--val-fraction", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-scan", type=int, default=6_000_000)
    ap.add_argument("--local", action="append")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    bos, eos = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]")
    if tok.get_vocab_size() > 65535:
        raise SystemExit("vocab > 65535: uint16 token files cannot hold it")
    rng, local = random.Random(args.seed), bc.parse_local(args.local)
    row_len = args.ctx + 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    budgets = {"function": args.tokens_functions, "stackoverflow": args.tokens_so if args.so_jsonl else 0,
               "commit": args.tokens_commits, "contest": args.tokens_contests}
    iters = {}
    if budgets["function"] > 0:
        iters["function"] = function_stream(local, rng, args.seed, args.max_scan)
    if budgets["stackoverflow"] > 0:
        so = load_so(args.so_jsonl, 5_000_000)
        print(f"stackoverflow: {len(so):,} usable pairs loaded", flush=True)
        iters["stackoverflow"] = cycle(so, rng, args.max_repeat)
    if budgets["commit"] > 0:
        cm = load_commits(local, args.max_scan)
        print(f"commits: {len(cm):,} small-edit pairs found", flush=True)
        iters["commit"] = cycle(cm, rng, args.max_repeat)
    if budgets["contest"] > 0:
        ct = load_contests(local, rng, args.max_scan)
        print(f"contests: {len(ct):,} pairs found", flush=True)
        iters["contest"] = cycle(ct, rng, args.max_repeat)

    packer = Packer(row_len, eos)
    train_w, val_w = Writer(args.out, "train"), Writer(args.out, "val")
    used = {k: 0 for k in iters}
    n_pairs = {k: 0 for k in iters}
    skipped_long = 0
    real_tokens = resp_tokens = 0
    active = {k for k in iters if budgets[k] > 0}
    t0, next_report = time.time(), 50e6

    def emit(rows):
        for toks, mask, real in rows:
            (val_w if rng.random() < args.val_fraction else train_w).add(toks, mask)

    while active:
        kinds = sorted(active)
        k = rng.choices(kinds, weights=[max(budgets[x] - used[x], 1.0) for x in kinds])[0]
        batch = list(itertools.islice(iters[k], 64))
        if len(batch) < 64:
            active.discard(k)                      # that source ran out (or hit max_repeat)
        if not batch:
            continue
        for (p_ids, r_ids), pair in zip(encode_pairs(tok, bos, eos, batch), batch):
            n = len(p_ids) + len(r_ids)
            if n > row_len:
                skipped_long += 1
                continue
            emit(packer.add(p_ids + r_ids, [0] * len(p_ids) + [1] * len(r_ids)))
            used[k] += n
            n_pairs[k] += 1
            real_tokens += n
            resp_tokens += len(r_ids)
        if used[k] >= budgets[k]:
            active.discard(k)
        if real_tokens >= next_report:
            next_report += 50e6
            print(f"  {real_tokens/1e6:,.0f}M tokens packed ({real_tokens/(time.time()-t0)/1e3:,.0f}k tok/s) | "
                  + " ".join(f"{x}={human(used[x])}" for x in sorted(used)), flush=True)
    emit(packer.flush())
    train_w.close()
    val_w.close()

    rows = train_w.rows + val_w.rows
    fill = real_tokens / max(rows * row_len, 1)
    meta = {"ctx": args.ctx, "rows_train": train_w.rows, "rows_val": val_w.rows, "real_tokens": real_tokens,
            "response_tokens": resp_tokens, "fill_rate": fill, "skipped_too_long": skipped_long,
            "tokens_per_kind": used, "pairs_per_kind": n_pairs, "budgets": budgets}
    Path(f"{args.out}.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\npacked {real_tokens:,} real tokens into {rows:,} rows of {row_len} (fill {fill:.1%}); "
          f"{resp_tokens / real_tokens:.0%} of tokens carry loss; {skipped_long:,} examples skipped (too long)")
    for k in sorted(used):
        short = "" if used[k] >= 0.98 * budgets[k] else f"   <-- short of its {human(budgets[k])} budget"
        print(f"  {k:<14}{human(used[k]):>10} tokens  {n_pairs[k]:>10,} pairs{short}")
    print(f"train rows {train_w.rows:,} | val rows {val_w.rows:,} | files: {args.out}.{{train,val}}.{{tokens,mask}}.bin")


if __name__ == "__main__":
    main()
