from __future__ import annotations

import dataclasses
import logging
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_project.config import ModelConfig, TrainConfig
from transformer_project.training.checkpoint import load_checkpoint, save_checkpoint
from transformer_project.training.logger import MetricsTracker
from transformer_project.training.optim import build_optimizer, get_lr

logger = logging.getLogger(__name__)

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class Trainer:
    """
    Encapsulates the full training loop: optimizer/scheduler, AMP, gradient
    accumulation, gradient clipping, periodic eval, checkpointing, and
    metrics logging. This is a class (not a 200-line flat script cell) so it
    can be unit tested and reused (e.g. called from a notebook, a CLI, or a
    different orchestrator) without copy-pasting the loop.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.device = device

        self.optimizer = build_optimizer(self.model, train_cfg)

        if train_cfg.amp_dtype not in _AMP_DTYPES:
            raise ValueError(f"Unknown amp_dtype: {train_cfg.amp_dtype}")
        self.amp_dtype = _AMP_DTYPES[train_cfg.amp_dtype]
        self.use_amp = train_cfg.use_amp and device.type == "cuda" and train_cfg.amp_dtype != "fp32"
        # GradScaler is only meaningful for fp16 (bf16 has enough dynamic
        # range that it isn't needed). The device-agnostic torch.amp.GradScaler
        # constructor only exists from torch>=2.3; on 2.2 we fall back to the
        # older torch.cuda.amp.GradScaler, which still works identically here
        # since this project only ever runs the scaler on CUDA.
        scaler_enabled = self.use_amp and self.amp_dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler(device="cuda", enabled=scaler_enabled)
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        self.metrics = MetricsTracker(
            use_wandb=train_cfg.use_wandb,
            wandb_project=train_cfg.wandb_project,
            wandb_run_name=train_cfg.wandb_run_name,
            config={"model": dataclasses.asdict(model_cfg), "train": dataclasses.asdict(train_cfg)},
        )

        self.step = 0
        self.best_val_loss = float("inf")
        self.ckpt_dir = str(Path(train_cfg.output_dir) / "checkpoints")

        if train_cfg.resume_from:
            ckpt = load_checkpoint(train_cfg.resume_from, self.model, self.optimizer, self.scaler)
            self.step = ckpt["step"]
            self.best_val_loss = ckpt["val_loss"]
            logger.info("Resumed from step %d", self.step)
        elif train_cfg.init_from:
            # SFT: load pretrained WEIGHTS only. Optimizer and step counter
            # stay fresh (step=0, new AdamW moment estimates) — this is a
            # new training phase building on the pretrained model, not a
            # continuation of the original pretraining run's step count.
            load_checkpoint(train_cfg.init_from, self.model)
            logger.info(
                "Initialized model weights from %s (fresh optimizer/step — this is SFT, not a resume)",
                train_cfg.init_from,
            )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, max_batches: int = 50) -> tuple[float, float]:
        self.model.eval()
        total, count = 0.0, 0
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                _, loss, _ = self.model(x, targets=y)
            total += loss.item()
            count += 1
        self.model.train()
        avg = total / max(count, 1)
        ppl = math.exp(min(avg, 20))  # capped to avoid float overflow on bad early checkpoints
        return avg, ppl

    def _run_micro_batch(self, train_iter):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(self.train_loader)
            x, y = next(train_iter)
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
            _, loss, _ = self.model(x, targets=y)
            loss = loss / self.cfg.grad_accum
        self.scaler.scale(loss).backward()
        return train_iter, loss.item()

    def train(self) -> None:
        self.model.train()

        if self.step > 0:
            # The DataLoader is rebuilt fresh on every process start, with the
            # SAME seed every time (torch.manual_seed(cfg.train.seed) runs
            # once at startup) — so shuffle=True produces an IDENTICAL batch
            # order on resume as it did originally. Without this fast-forward,
            # a resumed run would silently re-consume the same early batches
            # under new step numbers (e.g. step 1001 would get what was
            # actually batch #1's data), rather than continuing from where
            # training had actually gotten to in the data stream.
            #
            # This assumes batch_size and grad_accum are unchanged from the
            # run that produced this checkpoint — changing either between
            # resumes breaks the batch-count alignment this relies on.
            #
            # Skipping is done through a num_workers=0 DataLoader over the
            # SAME dataset/sampler config, not the multi-worker one the config
            # requests. Shuffle order depends only on the Sampler + fixed
            # seed, not on num_workers, so this produces an identical skip
            # sequence — but each next() avoids the multi-worker IPC
            # round-trip, which is what made this take many minutes rather
            # than being genuinely "cheap" as originally assumed. This
            # single-process loader is then kept for the rest of training
            # too (see below) since a fresh multi-worker iterator would lose
            # the position we just fast-forwarded to.
            already_consumed_batches = self.step * self.cfg.grad_accum
            logger.info(
                "Resuming at step %d: fast-forwarding past %d already-consumed "
                "micro-batches using a single-process loader (faster skip, "
                "but see log lines below for progress — this can still take "
                "a few minutes at this scale)",
                self.step, already_consumed_batches,
            )
            fast_loader = DataLoader(
                self.train_loader.dataset, batch_size=self.train_loader.batch_size,
                shuffle=True, num_workers=0, drop_last=True,
            )
            train_iter = iter(fast_loader)
            skipped = 0
            log_every_n = max(already_consumed_batches // 20, 10_000)  # ~20 progress lines total
            t_skip_start = time.time()
            while skipped < already_consumed_batches:
                try:
                    next(train_iter)
                except StopIteration:
                    train_iter = iter(fast_loader)
                    continue
                skipped += 1
                if skipped % log_every_n == 0:
                    elapsed = time.time() - t_skip_start
                    rate = skipped / elapsed if elapsed > 0 else 0
                    remaining = already_consumed_batches - skipped
                    eta_sec = remaining / rate if rate > 0 else float("inf")
                    logger.info(
                        "  fast-forward progress: %d/%d (%.1f%%) — %.0f batches/sec, ETA %.0fs",
                        skipped, already_consumed_batches,
                        100 * skipped / already_consumed_batches, rate, eta_sec,
                    )
            logger.info("Fast-forward complete in %.1fs — resuming real training.", time.time() - t_skip_start)
        else:
            train_iter = iter(self.train_loader)

        t0 = time.time()

        while self.step < self.cfg.max_steps:
            lr = get_lr(self.step, self.cfg)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(self.cfg.grad_accum):
                train_iter, micro_loss = self._run_micro_batch(train_iter)
                step_loss += micro_loss

            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step += 1
            elapsed = time.time() - t0
            t0 = time.time()

            if self.step % self.cfg.log_every == 0:
                self.metrics.log_step(
                    self.step, self.cfg.max_steps,
                    {"train_loss": step_loss, "lr": lr, "grad_norm": float(grad_norm)},
                    elapsed,
                )
                self.metrics.log(self.step, {"train_loss": step_loss, "lr": lr})

            if self.step % self.cfg.eval_every == 0:
                val_loss, val_ppl = self.evaluate(self.val_loader)
                self.metrics.log(self.step, {"val_loss": val_loss, "val_ppl": val_ppl})
                logger.info("  eval @ step %d: val_loss=%.4f val_ppl=%.2f", self.step, val_loss, val_ppl)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save("best")

            if self.step % self.cfg.save_every == 0:
                self._save("latest")

        self._save("final")
        best = self.metrics.best("val_loss")
        if best:
            logger.info("Training complete. Best val_loss=%.4f at step %s", best["val_loss"], best["step"])

    def _save(self, tag: str) -> str:
        val_loss = self.best_val_loss if self.best_val_loss != float("inf") else -1.0
        return save_checkpoint(
            self.model, self.optimizer, self.scaler, self.step, val_loss,
            dataclasses.asdict(self.model_cfg), dataclasses.asdict(self.cfg),
            self.ckpt_dir, tag=tag, keep_last_n=self.cfg.keep_last_n_checkpoints,
        )
