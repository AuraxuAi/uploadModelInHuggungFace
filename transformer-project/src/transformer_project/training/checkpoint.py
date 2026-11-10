"""
Checkpoint save/load.

Fixes vs. the original notebook:
  * Atomic writes: save to a temp file then os.replace() into place, so a
    crash mid-save (disk full, OOM-kill, preemption) can never leave a
    truncated/corrupt checkpoint at the path other code expects to load.
  * Rotation: keeps only the last N checkpoints plus a separate "best"
    checkpoint, so a long run doesn't silently fill the disk.
  * asdict() applied to dataclasses at the call site (dataclasses.asdict),
    not assumed — this module just deals with plain dicts so it doesn't
    need to import the config dataclasses at all.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    val_loss: float,
    model_cfg_dict: Dict[str, Any],
    train_cfg_dict: Dict[str, Any],
    ckpt_dir: str,
    tag: str = "latest",
    keep_last_n: int = 3,
) -> str:
    ckpt_dir_path = Path(ckpt_dir)
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir_path / f"ckpt_{tag}.pt"

    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "val_loss": val_loss,
        "model_cfg": model_cfg_dict,
        "train_cfg": train_cfg_dict,
    }

    # Atomic write: write to a temp file in the same directory (so os.replace
    # is a same-filesystem rename, not a copy), then swap it into place.
    fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir_path, suffix=".tmp")
    os.close(fd)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
    logger.info("Saved checkpoint: %s (step=%d, val_loss=%.4f)", final_path, step, val_loss)

    if tag == "latest":
        _rotate_step_checkpoints(ckpt_dir_path, step, payload, keep_last_n)

    return str(final_path)


def _rotate_step_checkpoints(ckpt_dir: Path, step: int, payload: dict, keep_last_n: int) -> None:
    """Also keep a step-numbered copy for rollback, pruning old ones."""
    step_path = ckpt_dir / f"ckpt_step{step}.pt"
    fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir, suffix=".tmp")
    os.close(fd)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, step_path)

    step_ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"), key=lambda p: p.stat().st_mtime)
    for old in step_ckpts[:-keep_last_n] if keep_last_n > 0 else []:
        old.unlink(missing_ok=True)
        logger.debug("Pruned old checkpoint: %s", old)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location or "cpu")
    state = ckpt["model"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # torch.compile() wraps parameters with an "_orig_mod." prefix.
        stripped = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(stripped, strict=False)
        logger.warning("Loaded checkpoint after stripping '_orig_mod.' prefix (torch.compile artifact).")

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    logger.info("Loaded checkpoint: step=%d val_loss=%.4f", ckpt["step"], ckpt["val_loss"])
    return ckpt
