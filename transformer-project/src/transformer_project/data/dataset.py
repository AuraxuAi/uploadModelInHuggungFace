"""
Dataset construction: tokenize raw sources once, persist to a memory-mapped
.bin file on disk, and serve fixed-length chunks from it.

This replaces the notebook's "Session Recovery" cell, which re-tokenized
everything from scratch every session unless you remembered to manually
copy .bin files to/from Google Drive. Here, `build_or_load_corpus` is
idempotent: if the .bin + manifest already exist and match the requested
config, it just memory-maps them (near-instant, low RAM) instead of
re-tokenizing.

Data provenance: every corpus build writes a `manifest.json` recording
exactly which source files/datasets went in, their sizes, and a content
hash. Silently training on unvetted scraped code (as the original notebook
did, cloning GitHub repos with no license filtering) is a legal and
reproducibility risk — at minimum you should be able to answer "what did
this model train on?" after the fact. This module makes that answerable;
it does NOT do license classification for you. If you plan to source
training data from other people's repositories, add an explicit allowlist
of licenses you've checked (e.g. MIT/BSD/Apache-2.0) before feeding files
in here.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from transformer_project.data.tokenizer import TokenizerWrapper

logger = logging.getLogger(__name__)


class TextChunkDataset(Dataset):
    """Serves fixed-length (x, y) next-token-prediction chunks from a token array."""

    def __init__(self, token_ids: np.memmap | np.ndarray, ctx: int):
        if len(token_ids) <= ctx:
            raise ValueError(
                f"Corpus has only {len(token_ids)} tokens, need more than context_len={ctx}."
            )
        self.data = token_ids
        self.ctx = ctx
        self.n = (len(self.data) - 1) // ctx
        logger.info("Dataset: %s tokens -> %s chunks of %d", f"{len(self.data):,}", f"{self.n:,}", ctx)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        start = idx * self.ctx
        x = torch.from_numpy(np.array(self.data[start: start + self.ctx], dtype=np.int64))
        y = torch.from_numpy(np.array(self.data[start + 1: start + self.ctx + 1], dtype=np.int64))
        return x, y


class SFTPairDataset(Dataset):
    """Serves (prompt, response) pairs for supervised fine-tuning, with loss
    masked (-100) on the prompt portion — the model is only trained to
    predict the response tokens, not to reproduce the prompt itself. This
    is what actually teaches "given this signature, continue THIS function"
    behavior, which plain next-token pretraining on file-level text does not
    reliably teach (see the repeated boilerplate-completion issue this was
    built to address).

    Expects pairs as a list of (prompt_ids, response_ids) already-tokenized
    integer lists. Sequences longer than ctx are truncated from the left of
    the prompt (keeping the full response, since that's what loss is
    computed on); shorter ones are left-padded with the tokenizer's pad/eos
    id, also masked out of the loss.
    """

    def __init__(self, pairs: list[tuple[list[int], list[int]]], ctx: int, pad_id: int):
        self.pairs = pairs
        self.ctx = ctx
        self.pad_id = pad_id
        logger.info("SFT dataset: %s (prompt, response) pairs, context_len=%d", f"{len(pairs):,}", ctx)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        prompt_ids, response_ids = self.pairs[idx]
        full = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + list(response_ids)

        if len(full) > self.ctx + 1:
            # Truncate from the LEFT of the prompt — preserve the full
            # response and as much prompt context as still fits, rather
            # than losing the end of the response (what we're training on).
            overflow = len(full) - (self.ctx + 1)
            full = full[overflow:]
            labels = labels[overflow:]
        else:
            pad_len = (self.ctx + 1) - len(full)
            full = full + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        x = torch.tensor(full[:-1], dtype=torch.int64)
        y = torch.tensor(labels[1:], dtype=torch.int64)
        return x, y


class PackedSFTDataset(Dataset):
    """Pre-packed fine-tuning data: many short (prompt, response) examples per sequence.

    Padding every example to the full context wastes most of the compute when examples are short
    (a 500-token example in a 2048 window = 75% padding). Instead scripts/build_sft_packed.py packs
    several examples into each (ctx+1)-token row and writes two parallel flat files:

        <prefix>.tokens.bin   token ids (uint16)
        <prefix>.mask.bin     1 where the token is part of a RESPONSE (loss), 0 for prompts / padding

    Labels are the next token wherever the NEXT token is a response token, else -100 - the same
    convention as SFTPairDataset (the first response token is predicted from the last prompt token).
    As in ordinary pretraining packing, examples in one row can attend to each other; every example
    starts with [BOS] and ends with [EOS].
    """

    def __init__(self, tokens_path: str, mask_path: str, ctx: int, token_dtype=np.uint16):
        self.ctx = ctx
        self.row = ctx + 1
        self.tokens = np.memmap(tokens_path, dtype=token_dtype, mode="r")
        self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
        if len(self.tokens) != len(self.mask) or len(self.tokens) % self.row:
            raise ValueError(f"{tokens_path}/{mask_path}: sizes do not match a whole number of rows of {self.row} tokens")
        self.n = len(self.tokens) // self.row
        logger.info("Packed SFT dataset: %s rows of %d tokens (%s tokens, %.0f%% of them in responses)",
                    f"{self.n:,}", self.row, f"{len(self.tokens):,}",
                    100.0 * float(np.asarray(self.mask[: min(len(self.mask), 5_000_000)]).mean()))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        a, b = i * self.row, (i + 1) * self.row
        t = np.asarray(self.tokens[a:b], dtype=np.int64)
        m = np.asarray(self.mask[a:b], dtype=np.int64)
        x = t[:-1]
        y = np.where(m[1:] == 1, t[1:], -100)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def encode_sft_pair(tokenizer: TokenizerWrapper, prompt: str, response: str) -> tuple[list[int], list[int]]:
    """Token ids for one fine-tuning example, laid out EXACTLY like a pretraining document:

        [BOS] prompt-tokens | response-tokens [EOS]
                            ^ loss starts here

    `tokenizer.encode()` wraps every text as [BOS] ... [EOS], so calling it on the prompt and on
    the response separately would put an [EOS][BOS] pair between them - a pattern that never occurs
    in the pretraining data and never at inference time. Here the raw tokens are used (no
    automatic special tokens): one [BOS] in front, and one [EOS] at the very end so the model
    learns where an answer stops.
    """
    def raw(text: str) -> list[int]:
        return tokenizer.tok.encode(text, add_special_tokens=False).ids

    prompt_ids = ([tokenizer.bos_id] if tokenizer.bos_id is not None else []) + raw(prompt)
    response_ids = raw(response) + ([tokenizer.eos_id] if tokenizer.eos_id is not None else [])
    return prompt_ids, response_ids


class MixedDataset(Dataset):
    """Fine-tuning pairs plus a share of ordinary pretraining chunks ("replay").

    Fine-tuning on a narrow set of examples makes a model drift away from what it learned in
    pretraining. Mixing some of the original data back in (loss on every token, as in pretraining)
    keeps those abilities. `replay_fraction` is the share of ALL examples that are replay chunks:
    0.25 means 3 fine-tuning pairs for every 1 pretraining chunk. The replay chunks are a fixed
    random subset of the corpus, chosen once from `seed`.
    """

    def __init__(self, sft_ds: Dataset, replay_ds: Dataset, replay_fraction: float, seed: int = 0):
        if not 0.0 < replay_fraction < 1.0:
            raise ValueError("replay_fraction must be strictly between 0 and 1")
        self.sft, self.replay = sft_ds, replay_ds
        self.n_sft = len(sft_ds)
        want = int(round(self.n_sft * replay_fraction / (1.0 - replay_fraction)))
        want = min(want, len(replay_ds))
        rng = np.random.default_rng(seed)
        self.replay_idx = rng.choice(len(replay_ds), size=want, replace=False)
        logger.info("Mixed dataset: %s fine-tuning pairs + %s pretraining chunks (%.0f%% replay)",
                    f"{self.n_sft:,}", f"{want:,}", 100.0 * want / max(self.n_sft + want, 1))

    def __len__(self) -> int:
        return self.n_sft + len(self.replay_idx)

    def __getitem__(self, i: int):
        if i < self.n_sft:
            return self.sft[i]
        return self.replay[int(self.replay_idx[i - self.n_sft])]


@dataclass
class CorpusManifest:
    sources: List[str]
    num_tokens: int
    tokenizer_path: str
    content_hash: str

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "CorpusManifest":
        with open(path, "r") as f:
            return cls(**json.load(f))


def _hash_sources(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode())
        stat = Path(p).stat()
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()[:16]


def tokenize_files_to_bin(
    files: List[str],
    tokenizer: TokenizerWrapper,
    out_bin: str,
    out_manifest: str,
    eos_between_files: bool = True,
    max_tokens: int | None = None,
    file_sep_text: str | None = None,
) -> np.memmap:
    """Tokenize a list of local files into a single memory-mapped uint32 array.

    max_tokens: stop once this many tokens have been written, so a source
    can be capped to an exact token budget (e.g. for a fixed data-mix
    ratio) instead of tokenizing every file regardless of size.

    file_sep_text: if given (e.g. "<file_sep>"), this literal text is
    encoded and inserted before each file's tokens, StarCoder2/Qwen2.5-Coder
    style, so the model has an explicit signal for "a new file starts here"
    instead of having to infer file boundaries purely from content. Train
    the tokenizer on a corpus that already contains this string (or add it
    as an explicit added/special token) so it doesn't get fragmented into
    multiple sub-word pieces.
    """
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} source file(s) missing, e.g. {missing[:3]}")

    out_bin_path = Path(out_bin)
    out_manifest_path = Path(out_manifest)
    content_hash = _hash_sources(files)

    if out_bin_path.exists() and out_manifest_path.exists():
        existing = CorpusManifest.from_json(out_manifest_path)
        if existing.content_hash == content_hash and existing.tokenizer_path == str(tokenizer):
            logger.info("Corpus cache hit: reusing %s (%s tokens)", out_bin, f"{existing.num_tokens:,}")
            return np.memmap(out_bin_path, dtype=np.uint32, mode="r")

    sep_ids = tokenizer.encode(file_sep_text) if file_sep_text else []
    out_bin_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream both the READ and the WRITE, instead of f.read()-ing a whole
    # file and tokenizer.encode()-ing it in one call, then accumulating
    # every token in one Python list. A single combined 69MB file (like
    # all_java_code.txt) still becomes ~17M+ token IDs in that old pattern
    # — the exact OOM crash we already fixed for the HF streaming path,
    # resurfacing here because this function historically assumed many
    # small files, not one huge concatenated one.
    #
    # If a file was built with file_sep_text markers already in it (e.g.
    # combine_java_files.py's <file_sep> boundaries), split and encode it
    # section-by-section instead of as one giant string, so peak memory
    # stays bounded regardless of overall file size.
    FLUSH_EVERY = 500_000
    buffer: List[int] = []
    total_tokens = 0
    used_files: List[str] = []

    def flush(buf: List[int], f):
        nonlocal total_tokens
        chunk = np.array(buf, dtype=np.uint32)
        f.write(chunk.tobytes())
        total_tokens += len(chunk)

    with open(out_bin_path, "wb") as out_f:
        for fp in files:
            if max_tokens is not None and total_tokens + len(buffer) >= max_tokens:
                break

            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            # Split on the same separator text this function would insert,
            # so a pre-combined multi-document file is encoded in small
            # pieces rather than as one massive string.
            sections = text.split(file_sep_text) if file_sep_text and file_sep_text in text else [text]

            for section in sections:
                if not section:
                    continue
                if max_tokens is not None and total_tokens + len(buffer) >= max_tokens:
                    break
                if sep_ids:
                    buffer.extend(sep_ids)
                buffer.extend(tokenizer.encode(section))
                if eos_between_files and tokenizer.eos_id is not None:
                    buffer.append(tokenizer.eos_id)

                if len(buffer) >= FLUSH_EVERY:
                    flush(buffer, out_f)
                    buffer = []

            used_files.append(fp)

        if max_tokens is not None and len(buffer) + total_tokens > max_tokens:
            buffer = buffer[: max_tokens - total_tokens]
        if buffer:
            flush(buffer, out_f)

    logger.info("Tokenized %d/%d files -> %s tokens, cached at %s",
                len(used_files), len(files), f"{total_tokens:,}", out_bin)
    manifest = CorpusManifest(
        sources=used_files, num_tokens=total_tokens, tokenizer_path=str(tokenizer), content_hash=content_hash
    )
    manifest.to_json(out_manifest_path)
    return np.memmap(out_bin_path, dtype=np.uint32, mode="r")


def tokenize_hf_dataset_to_bin(
    name: str,
    split: str,
    text_column: str,
    tokenizer: TokenizerWrapper,
    out_bin: str,
    max_samples: int = 10_000_000_000,
    max_tokens: int | None = None,
    log_every: int = 10_000,
    data_dir: str | None = None,
    dataset_config: str | None = None,
) -> np.memmap:
    """Stream + tokenize a HuggingFace dataset without loading it fully into RAM.

    data_dir restricts to a subdirectory of the dataset, e.g. "data/python"
    for bigcode/the-stack-dedup, which is organized one directory per
    language rather than exposed as separate named configs.

    dataset_config selects a *builder config* instead, e.g. "20231101.en"
    for wikimedia/wikipedia, which is organized as one config per
    (dump date, language) rather than as subdirectories. data_dir and
    dataset_config are different HF mechanisms — set only the one your
    dataset actually uses, not both.

    max_tokens: stop once this many tokens have been written — the way to
    hit an exact token budget for one ingredient of a data mix (e.g. "100M
    tokens of Wikipedia") instead of guessing a sample count and hoping it
    lands close. Checked far more often than max_samples, so the cutoff is
    tight rather than overshooting by a whole flush chunk.
    """
    from datasets import load_dataset  # imported lazily; optional dependency path

    out_bin_path = Path(out_bin)
    if out_bin_path.exists():
        logger.info("Corpus cache hit: reusing %s", out_bin)
        return np.memmap(out_bin_path, dtype=np.uint32, mode="r")

    ds = load_dataset(name, dataset_config, data_dir=data_dir, split=split, streaming=True)
    out_bin_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream straight to disk in bounded chunks instead of accumulating every
    # token as a Python int in one giant list. A list of ~350M+ Python ints
    # easily uses 10-15GB+ of RAM by itself, then np.array(all_ids, ...) tries
    # to allocate a second full-size buffer on top of that — on a ~12-13GB
    # Colab runtime this silently OOM-kills the process right as the loop
    # finishes, with no traceback, which looks like a random hang/disconnect.
    FLUSH_EVERY = 500_000  # tokens buffered in RAM before each disk write
    buffer: List[int] = []
    total_tokens = 0

    with open(out_bin_path, "wb") as f:
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            if max_tokens is not None and total_tokens + len(buffer) >= max_tokens:
                break
            buffer.extend(tokenizer.encode(sample[text_column]))
            if tokenizer.eos_id is not None:
                buffer.append(tokenizer.eos_id)

            if len(buffer) >= FLUSH_EVERY:
                chunk = np.array(buffer, dtype=np.uint32)
                f.write(chunk.tobytes())
                total_tokens += len(chunk)
                buffer = []  # drop the old buffer instead of growing it forever

            if i % log_every == 0 and i > 0:
                logger.info(
                    "Loaded %s samples (%.1fM tokens)",
                    f"{i:,}", (total_tokens + len(buffer)) / 1e6,
                )

        if max_tokens is not None and len(buffer) + total_tokens > max_tokens:
            buffer = buffer[: max_tokens - total_tokens]
        if buffer:  # flush whatever's left under FLUSH_EVERY
            chunk = np.array(buffer, dtype=np.uint32)
            f.write(chunk.tobytes())
            total_tokens += len(chunk)

    logger.info("Tokenized HF dataset %s -> %s tokens, cached at %s", name, f"{total_tokens:,}", out_bin)
    return np.memmap(out_bin_path, dtype=np.uint32, mode="r")


def split_train_val(token_ids: np.memmap, val_fraction: float):
    split_idx = int(len(token_ids) * (1 - val_fraction))
    return token_ids[:split_idx], token_ids[split_idx:]
