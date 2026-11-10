from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional


def setup_logging(level: str = "INFO") -> None:
    """Configure stdlib logging once, for the whole process. Call this at the
    top of any entry point (train script, generate script, etc.) instead of
    relying on bare `print()` calls scattered through the codebase."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # These libraries log one INFO line per HTTP request/chunk during dataset
    # streaming (httpx/httpcore) or filesystem locking (filelock). At the
    # sample counts a real training run uses, that floods stdout badly enough
    # to make Colab's frontend feel like it's hung. Keep them at WARNING
    # regardless of our own --log-level, since their INFO output is noise for
    # this project's purposes, not signal.
    for noisy in ("httpx", "httpcore", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class MetricsTracker:
    """Tracks training metrics and optionally mirrors them to Weights & Biases.

    Unlike the notebook version, a W&B init failure here is a warning, not a
    silent swallow that leaves you debugging "why isn't anything logging" —
    it's logged clearly via the standard logger.
    """

    def __init__(
        self,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.history: List[Dict[str, Any]] = []
        self.use_wandb = False
        self._wandb = None

        if use_wandb:
            try:
                import wandb

                wandb.init(project=wandb_project, name=wandb_run_name, config=config or {})
                self._wandb = wandb
                self.use_wandb = True
                self.logger.info("W&B initialized (project=%s, run=%s)", wandb_project, wandb_run_name)
            except Exception as e:  # noqa: BLE001 - deliberately broad, this is optional infra
                self.logger.warning("W&B init failed, continuing without it: %s", e)

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        self.history.append({"step": step, **metrics})
        if self.use_wandb:
            self._wandb.log(metrics, step=step)

    def log_step(self, step: int, max_steps: int, metrics: Dict[str, Any], elapsed_s: float) -> None:
        parts = [f"step {step:>6d}/{max_steps}"]
        for k, v in metrics.items():
            if k == "lr" and isinstance(v, float):
                parts.append(f"{k}={v:.3e}")          # %.4f would print 0.0000 for the whole warmup
            else:
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        parts.append(f"[{elapsed_s:.2f}s/step]")
        self.logger.info("  ".join(parts))

    def best(self, key: str = "val_loss") -> Optional[Dict[str, Any]]:
        candidates = [h for h in self.history if key in h]
        if not candidates:
            return None
        return min(candidates, key=lambda h: h[key])
