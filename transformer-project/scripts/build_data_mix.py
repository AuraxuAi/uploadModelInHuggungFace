"""
Blend several already-tokenized .bin sources into one shuffled training
corpus at fixed token ratios, instead of training on one source fully
before moving to the next (which risks catastrophic forgetting — see the
conversation this came out of: mixing must happen from step one, not
sequentially by domain).

Usage:
    python build_data_mix.py --out artifacts/mix_50m/corpus.bin \
        --source raw_code.bin:0.70 \
        --source wiki.bin:0.10 \
        --source math.bin:0.10 \
        --source repo_java.bin:0.10 \
        --context-len 512 --seed 42

Two ways to size the output:

  * default (no --total-tokens): the corpus is as large as the source that is
    SMALLEST relative to its ratio allows, and nothing is repeated. One tiny
    source with a big ratio therefore shrinks the WHOLE corpus - check the
    "Limiting source" line it prints.
  * --total-tokens N [--max-repeat R]: produce exactly N tokens at the given
    ratios; a source that cannot supply its share is repeated (reshuffled each
    pass), up to R times its size, otherwise the script stops with an error.
    This is what the Colab notebook uses, with per-source ratios computed from
    the tokens actually available.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def parse_source_arg(arg: str) -> tuple[str, float]:
    path, ratio = arg.rsplit(":", 1)
    return path, float(ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                     help="path:target_ratio, repeatable, e.g. raw_code.bin:0.70")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context-len", type=int, default=512,
                     help="chunk size in tokens — should match model.context_len")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total-tokens", type=int, default=0,
                    help="exact size of the output corpus in tokens (enables repeating small sources)")
    ap.add_argument("--max-repeat", type=float, default=1.0,
                    help="with --total-tokens: most times a source may be repeated (1.0 = never)")
    ap.add_argument("--dtype", choices=["uint16", "uint32"], default="uint32",
                     help="dtype of BOTH the input .bin files and the output corpus. "
                          "Use uint16 for vocab <= 65535 (halves disk use).")
    args = ap.parse_args()

    DT = np.dtype(args.dtype)
    sources = [parse_source_arg(s) for s in args.source]
    total_ratio = sum(r for _, r in sources)
    if abs(total_ratio - 1.0) > 1e-3:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    chunk_size = args.context_len
    per_source_chunks: dict[str, list[int]] = {}
    per_source_tokens: dict[str, int] = {}

    for path, ratio in sources:
        arr = np.memmap(path, dtype=DT, mode="r")
        n_chunks = len(arr) // chunk_size
        per_source_tokens[path] = len(arr)
        per_source_chunks[path] = list(range(n_chunks))
        print(f"{path}: {len(arr):,} tokens -> {n_chunks:,} available chunks (target ratio {ratio:.0%})")

    # Enforce the target ratios instead of just dumping every available chunk
    # from every source (which silently produces whatever ratio the raw file
    # sizes happen to give you — the bug that motivated this rewrite: a
    # 1.2M-token source and a 16K-token source do NOT average out to their
    # "target_ratio" labels on their own).
    #
    # total_output_chunks is capped by the SMALLEST source relative to its
    # own ratio, since we only repeat (not fabricate) data: a source can
    # supply at most (its own chunk count / its ratio) chunks worth of
    # total corpus before every other source would need to repeat past a
    # single pass. Sources with fewer chunks than their ratio needs are
    # upsampled (repeated, shuffled independently); sources with more are
    # subsampled once, un-repeated.
    if args.total_tokens:
        total_output_chunks = args.total_tokens // chunk_size
        print(f"\nRequested corpus: {total_output_chunks:,} chunks ({total_output_chunks * chunk_size:,} tokens)")
        over = []
        for path, ratio in sources:
            need = round(total_output_chunks * ratio)
            have = len(per_source_chunks[path])
            if have == 0 and need > 0 or (have and need / have > args.max_repeat + 1e-9):
                over.append(f"  {path}: needs {need:,} chunks but has {have:,} "
                            f"({need / max(have, 1):.1f}x; --max-repeat is {args.max_repeat})")
        if over:
            raise SystemExit("Some sources would be repeated more than --max-repeat allows:\n" + "\n".join(over)
                             + "\nLower their ratios (or raise --max-repeat).")
    else:
        limiting_source, limiting_scale = min(
            ((path, per_source_chunks[path].__len__() / ratio if ratio > 0 else float("inf"))
             for path, ratio in sources),
            key=lambda x: x[1],
        )
        total_output_chunks = int(limiting_scale)
        print(f"\nLimiting source: {limiting_source} (smallest relative to its target ratio)")
        print(f"Total output corpus: {total_output_chunks:,} chunks "
              f"({total_output_chunks * chunk_size:,} tokens)\n")

    rng = random.Random(args.seed)
    all_chunks: list[tuple[str, int]] = []

    for path, ratio in sources:
        n_needed = round(total_output_chunks * ratio)
        available = per_source_chunks[path]
        rng.shuffle(available)

        if n_needed <= len(available):
            chosen = available[:n_needed]
        else:
            # Upsample with repetition — cycle through the shuffled pool
            # as many times as needed, reshuffling each full pass so
            # repeats aren't in the same relative order every time.
            chosen = []
            while len(chosen) < n_needed:
                pool = available[:]
                rng.shuffle(pool)
                chosen.extend(pool)
            chosen = chosen[:n_needed]
            repeats = n_needed / len(available)
            print(f"  {path}: only {len(available)} chunks available for {n_needed} needed "
                  f"-> repeated ~{repeats:.1f}x (upsampling a small source this much reduces "
                  f"its value as *diverse* signal — treat this ratio's quality with that caveat)")

        for c in chosen:
            all_chunks.append((path, c * chunk_size))

    # Shuffle chunk order across ALL sources together — this is what makes
    # the mix "from the start" rather than "all of source A, then source B".
    rng.shuffle(all_chunks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = len(all_chunks) * chunk_size

    mm_out = np.memmap(out_path, dtype=DT, mode="w+", shape=(total_tokens,))
    open_arrays = {path: np.memmap(path, dtype=DT, mode="r") for path, _ in sources}

    # Group by source and process each source's chunks with vectorized NumPy
    # fancy-indexing instead of a pure-Python loop over every individual
    # chunk. The original per-chunk loop did ~5M+ tiny random-access memmap
    # slice operations one at a time — each carrying real Python interpreter
    # overhead on top of the actual I/O, which is what made this take over
    # an hour at this scale. Batching gathers/writes as one vectorized NumPy
    # call per batch removes that per-iteration Python overhead, typically
    # cutting this from ~90 minutes to low single-digit minutes.
    #
    # BATCH_CHUNKS bounds how many chunks' worth of data get materialized in
    # RAM per batch — same "flush periodically" pattern as the tokenizer
    # memory fixes elsewhere in this project — so this stays memory-safe
    # regardless of total corpus size, not just fast.
    BATCH_CHUNKS = 50_000  # 50k chunks * 512 tokens * 4 bytes = ~100MB per batch — safe on any Colab tier

    # output_position -> (path, start) is what all_chunks holds; group the
    # output positions by source so each source can be gathered in one shot.
    positions_by_source: dict[str, list[int]] = {path: [] for path, _ in sources}
    starts_by_source: dict[str, list[int]] = {path: [] for path, _ in sources}
    for out_pos, (path, start) in enumerate(all_chunks):
        positions_by_source[path].append(out_pos)
        starts_by_source[path].append(start)

    for path, _ in sources:
        positions = np.array(positions_by_source[path], dtype=np.int64)
        starts = np.array(starts_by_source[path], dtype=np.int64)
        src_arr = open_arrays[path]
        n = len(positions)

        for batch_start in range(0, n, BATCH_CHUNKS):
            batch_end = min(batch_start + BATCH_CHUNKS, n)
            batch_starts = starts[batch_start:batch_end]
            batch_positions = positions[batch_start:batch_end]

            # Vectorized gather: build a (num_chunks, chunk_size) index matrix
            # and fancy-index the source array in one call, instead of
            # slicing chunk_size elements out in a Python loop per chunk.
            idx_matrix = batch_starts[:, None] + np.arange(chunk_size)[None, :]
            gathered = src_arr[idx_matrix]  # shape: (num_chunks_in_batch, chunk_size)

            # Vectorized scatter into the output memmap at this batch's
            # (possibly non-contiguous, shuffled) output positions.
            out_idx_matrix = batch_positions[:, None] * chunk_size + np.arange(chunk_size)[None, :]
            mm_out[out_idx_matrix.ravel()] = gathered.ravel()

        print(f"  wrote {n:,} chunks from {path}")

    mm_out.flush()

    manifest = {
        "sources": {path: {"tokens": n, "target_ratio": ratio}
                    for (path, ratio), n in zip(sources, per_source_tokens.values())},
        "chunk_size": chunk_size,
        "total_chunks": len(all_chunks),
        "total_tokens": total_tokens,
        "seed": args.seed,
    }
    with open(out_path.with_suffix(".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {total_tokens:,} tokens ({len(all_chunks):,} chunks of {chunk_size}) to {out_path}")
    print(f"Manifest: {out_path.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
