"""
Command-line entry points, installed as console scripts by pyproject.toml:
  tp-train-tokenizer --files ... --vocab-size 32000 --out artifacts/tokenizer.json
  tp-train --config configs/t4.yaml
  tp-generate --config configs/t4.yaml --checkpoint artifacts/run/checkpoints/ckpt_best.pt --prompt "..."
  tp-export-hf --config configs/t4.yaml --checkpoint ... --out artifacts/hf_export

This replaces "edit the right notebook cell and re-run" with ordinary CLI
flags + a version-controlled YAML config, so a training run is reproducible
from a single command instead of a sequence of manual notebook edits.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformer_project.config import RunConfig
from transformer_project.data.dataset import (
    MixedDataset, PackedSFTDataset, SFTPairDataset, TextChunkDataset, encode_sft_pair, split_train_val, tokenize_files_to_bin, tokenize_hf_dataset_to_bin,
)
from transformer_project.data.tokenizer import TokenizerWrapper
from transformer_project.inference.generate import generate
from transformer_project.model.transformer import Transformer
from transformer_project.training.checkpoint import load_checkpoint
from transformer_project.training.logger import setup_logging
from transformer_project.training.trainer import Trainer

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenizer_main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--files", nargs="+", required=True, help="Text/code files to train on.")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--out", default="artifacts/tokenizer.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    TokenizerWrapper.train(args.files, vocab_size=args.vocab_size, save_path=args.out)


def _with_replay(cfg: RunConfig, train_ds):
    """Mix ordinary pretraining chunks into a fine-tuning dataset so the model does not forget."""
    if not cfg.data.replay_corpus:
        return train_ds
    import numpy as np
    replay_tokens = np.memmap(cfg.data.replay_corpus, dtype=getattr(np, cfg.data.token_dtype), mode="r")
    replay_train, _ = split_train_val(replay_tokens, 0.001)    # same held-out tail as pretraining
    replay_ds = TextChunkDataset(replay_train, cfg.model.context_len)
    return MixedDataset(train_ds, replay_ds, cfg.data.replay_fraction, seed=cfg.train.seed)


def _build_dataloaders(cfg: RunConfig, tokenizer: TokenizerWrapper):
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cfg.data.sft_packed_prefix:
        import numpy as np
        pre, dt = cfg.data.sft_packed_prefix, getattr(np, cfg.data.token_dtype)
        train_ds = PackedSFTDataset(f"{pre}.train.tokens.bin", f"{pre}.train.mask.bin", cfg.model.context_len, dt)
        val_ds = PackedSFTDataset(f"{pre}.val.tokens.bin", f"{pre}.val.mask.bin", cfg.model.context_len, dt)
        train_ds = _with_replay(cfg, train_ds)
        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                                  num_workers=cfg.train.num_workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                                num_workers=cfg.train.num_workers, drop_last=True)
        return train_loader, val_loader

    if len(cfg.data.train_files) == 1 and cfg.data.train_files[0].endswith(".jsonl"):
        # SFT pairs file (from scripts/extract_sft_pairs.py): each line is
        # {"prompt": ..., "response": ...}. Tokenized here (not pre-tokenized
        # to a .bin like pretraining) since pairs need to stay separable —
        # a flat token array loses the prompt/response boundary needed for
        # loss masking.
        import json
        pairs = []
        with open(cfg.data.train_files[0]) as f:
            for line in f:
                row = json.loads(line)
                pairs.append(encode_sft_pair(tokenizer, row["prompt"], row["response"]))
        logger.info("Loaded %s SFT (prompt, response) pairs from %s", f"{len(pairs):,}", cfg.data.train_files[0])

        pad_id = tokenizer.eos_id if tokenizer.eos_id is not None else 0
        split_idx = int(len(pairs) * (1 - cfg.data.val_fraction))
        train_ds = SFTPairDataset(pairs[:split_idx], cfg.model.context_len, pad_id)
        val_ds = SFTPairDataset(pairs[split_idx:], cfg.model.context_len, pad_id)

        train_ds = _with_replay(cfg, train_ds)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.train.num_workers, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.train.num_workers, drop_last=True,
        )
        return train_loader, val_loader

    if len(cfg.data.train_files) == 1 and cfg.data.train_files[0].endswith(".bin"):
        # Pre-built, pre-mixed corpus (e.g. from scripts/build_data_mix.py) —
        # already tokenized uint32 token IDs, blended from several sources
        # at fixed ratios. Load directly rather than re-tokenizing it as if
        # it were raw text, which would corrupt it.
        import numpy as np
        tokens = np.memmap(cfg.data.train_files[0], dtype=getattr(np, cfg.data.token_dtype), mode="r")
        logger.info("Loaded pre-built mixed corpus: %s (%s tokens)",
                    cfg.data.train_files[0], f"{len(tokens):,}")
    elif cfg.data.train_files:
        tokens = tokenize_files_to_bin(
            cfg.data.train_files, tokenizer,
            out_bin=str(cache_dir / "corpus.bin"), out_manifest=str(cache_dir / "manifest.json"),
        )
    elif cfg.data.hf_dataset_name:
        tokens = tokenize_hf_dataset_to_bin(
            cfg.data.hf_dataset_name, cfg.data.hf_dataset_split, cfg.data.hf_text_column,
            tokenizer, out_bin=str(cache_dir / "corpus.bin"), max_samples=cfg.data.hf_max_samples,
            data_dir=cfg.data.hf_data_dir, dataset_config=cfg.data.hf_dataset_config,
        )
    else:
        raise ValueError("DataConfig must set either train_files or hf_dataset_name.")

    train_tokens, val_tokens = split_train_val(tokens, cfg.data.val_fraction)
    train_ds = TextChunkDataset(train_tokens, cfg.model.context_len)
    val_ds = TextChunkDataset(val_tokens, cfg.model.context_len)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, drop_last=True,
    )
    return train_loader, val_loader


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train the transformer.")
    parser.add_argument("--config", required=True, help="Path to a YAML RunConfig.")
    parser.add_argument("--resume", default=None,
                         help="Path to a checkpoint (.pt) to resume from — restores model "
                              "weights, optimizer state, AMP scaler state, and the step "
                              "counter, so training continues from exactly where it left "
                              "off (same LR schedule position, same optimizer momentum) "
                              "rather than restarting from step 0.")
    parser.add_argument("--init-from", default=None,
                         help="Path to a checkpoint (.pt) to initialize model WEIGHTS from, "
                              "for SFT/fine-tuning — unlike --resume, the optimizer and step "
                              "counter start fresh. Use this to fine-tune a pretrained "
                              "checkpoint on a new (e.g. SFT) dataset/config, not to continue "
                              "the original pretraining run.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = RunConfig.from_yaml(args.config)
    if args.resume:
        cfg.train.resume_from = args.resume
    if args.init_from:
        cfg.train.init_from = args.init_from
    torch.manual_seed(cfg.train.seed)

    tokenizer = TokenizerWrapper.load(cfg.data.tokenizer_path)
    if tokenizer.vocab_size != cfg.model.vocab_size:
        logger.warning(
            "Tokenizer vocab_size (%d) != model.vocab_size in config (%d); using tokenizer's.",
            tokenizer.vocab_size, cfg.model.vocab_size,
        )
        cfg.model.vocab_size = tokenizer.vocab_size

    train_loader, val_loader = _build_dataloaders(cfg, tokenizer)
    model = Transformer(cfg.model)
    logger.info("Model: %.2fM parameters", model.num_parameters() / 1e6)

    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(Path(cfg.train.output_dir) / "resolved_config.yaml")

    trainer = Trainer(model, train_loader, val_loader, cfg.model, cfg.train, _device())
    trainer.train()


def generate_main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--strategy", choices=["greedy", "sample", "beam"], default="sample")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="1.0 = off (right for code). Values like 1.1 compound with every repeat and hurt code badly.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = RunConfig.from_yaml(args.config)
    device = _device()

    tokenizer = TokenizerWrapper.load(cfg.data.tokenizer_path)
    model = Transformer(cfg.model)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.to(device)

    text = generate(
        model, tokenizer, args.prompt, device,
        max_new_tokens=args.max_new_tokens, strategy=args.strategy,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(text)


def export_main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint to a real HF-compatible format.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="artifacts/hf_export")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = RunConfig.from_yaml(args.config)

    from transformer_project.model.hf_export import MiniTransformerForCausalLM

    # Load into a real Transformer first so we get the "_orig_mod." stripping
    # (torch.compile artifact) handled by load_checkpoint, then hand its
    # already-clean state_dict to the HF wrapper rather than re-parsing the
    # raw checkpoint file a second time.
    source_model = Transformer(cfg.model)
    load_checkpoint(args.checkpoint, source_model)
    hf_model = MiniTransformerForCausalLM.from_local_checkpoint(cfg.model, source_model.state_dict())
    hf_model.save_pretrained(args.out)
    logger.info("Exported HF-compatible model to %s", args.out)
