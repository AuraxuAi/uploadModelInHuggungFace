"""
Configuration objects for the model and the training run.

Design goals vs. the original notebook:
  * No hardcoded '/content' or '/gdrive' paths — everything is a plain
    argument with a sane default that works on any machine.
  * Configs are plain dataclasses that (de)serialize to YAML, so a run
    can be fully reproduced from a single config file checked into git,
    instead of "edit cell 4 and re-run".
  * Validation happens at construction time (__post_init__) instead of
    failing deep inside training with a confusing shape-mismatch error.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    context_len: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    d_model: int = 512
    d_ff_mult: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.05
    rope_base: int = 10000
    norm_eps: float = 1e-5
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.vocab_size <= 0 or self.context_len <= 0 or self.n_layers <= 0:
            raise ValueError("vocab_size, context_len, and n_layers must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        return int(self.d_ff_mult * self.d_model)


@dataclass
class DataConfig:
    """Where training data comes from. Exactly one source should be set."""
    train_files: List[str] = field(default_factory=list)      # local text/code files
    hf_dataset_name: Optional[str] = None                       # e.g. "roneneldan/TinyStories"
    hf_dataset_split: str = "train"
    hf_data_dir: Optional[str] = None                           # e.g. "data/python" for bigcode/the-stack-dedup
    hf_dataset_config: Optional[str] = None                     # e.g. "20231101.en" for wikimedia/wikipedia
    hf_text_column: str = "text"
    hf_max_samples: int = 100_000
    val_fraction: float = 0.05
    tokenizer_path: str = "artifacts/tokenizer.json"
    tokenizer_vocab_size: int = 32000
    cache_dir: str = "artifacts/cache"
    # dtype of pre-built .bin corpora. uint16 is valid while vocab_size <= 65535
    # and halves disk/page-cache use (20B tokens: 40 GB instead of 80 GB).
    token_dtype: str = "uint32"
    # Fine-tuning only (train_files = a .jsonl of prompt/response pairs): mix in ordinary pretraining
    # data so the model does not forget it. replay_corpus is a pre-built .bin (same token_dtype);
    # replay_fraction is the share of ALL training examples that come from it.
    # Packed fine-tuning data: prefix of the files written by scripts/build_sft_packed.py
    # (<prefix>.train.tokens.bin / .train.mask.bin / .val.tokens.bin / .val.mask.bin). Takes precedence
    # over train_files. replay_corpus / replay_fraction below can be combined with it.
    sft_packed_prefix: Optional[str] = None
    replay_corpus: Optional[str] = None
    replay_fraction: float = 0.0


    def __post_init__(self) -> None:
        if self.token_dtype not in {"uint16", "uint32"}:
            raise ValueError("token_dtype must be 'uint16' or 'uint32'")
        if self.replay_corpus and not 0.0 < self.replay_fraction < 1.0:
            raise ValueError("replay_corpus needs 0 < replay_fraction < 1")


@dataclass
class TrainConfig:
    output_dir: str = "artifacts/run"
    seed: int = 42

    batch_size: int = 16
    grad_accum: int = 4
    max_steps: int = 10_000
    warmup_steps: int = 200
    eval_every: int = 500
    save_every: int = 500
    log_every: int = 20

    lr: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    amp_dtype: str = "bf16"      # "bf16" | "fp16" | "fp32"
    use_amp: bool = True
    num_workers: int = 2

    use_wandb: bool = False
    wandb_project: str = "transformer-project"
    wandb_run_name: Optional[str] = None

    resume_from: Optional[str] = None
    init_from: Optional[str] = None  # SFT: load pretrained WEIGHTS only (fresh
                                       # optimizer, fresh step counter) — different
                                       # from resume_from, which restores full
                                       # training state to continue the SAME run
    keep_last_n_checkpoints: int = 3

    def __post_init__(self) -> None:
        if self.amp_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("amp_dtype must be one of: bf16, fp16, fp32")
        if self.max_steps <= self.warmup_steps:
            raise ValueError("max_steps must be greater than warmup_steps")


@dataclass
class RunConfig:
    """Top-level config bundling model + data + training settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(
                {
                    "model": asdict(self.model),
                    "data": asdict(self.data),
                    "train": asdict(self.train),
                },
                f,
                sort_keys=False,
            )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)
