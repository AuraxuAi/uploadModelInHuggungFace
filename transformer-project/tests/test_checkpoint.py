import torch

from transformer_project.config import ModelConfig, TrainConfig
from transformer_project.model.transformer import Transformer
from transformer_project.training.checkpoint import load_checkpoint, save_checkpoint
from transformer_project.training.optim import build_optimizer


def _tiny_setup():
    cfg = ModelConfig(vocab_size=32, context_len=8, n_layers=1, n_heads=2, n_kv_heads=1, d_model=16, d_ff_mult=2.0)
    model = Transformer(cfg)
    train_cfg = TrainConfig(max_steps=100, warmup_steps=10)
    optimizer = build_optimizer(model, train_cfg)
    return cfg, model, optimizer


def test_save_and_load_roundtrip(tmp_path):
    cfg, model, optimizer = _tiny_setup()
    ckpt_dir = tmp_path / "ckpts"

    path = save_checkpoint(
        model, optimizer, scaler=None, step=42, val_loss=1.23,
        model_cfg_dict=cfg.__dict__, train_cfg_dict={}, ckpt_dir=str(ckpt_dir), tag="latest",
    )
    assert (ckpt_dir / "ckpt_latest.pt").exists()

    # No leftover temp files after an atomic write.
    tmp_files = list(ckpt_dir.glob("*.tmp"))
    assert tmp_files == []

    new_model = Transformer(cfg)
    ckpt = load_checkpoint(path, new_model)
    assert ckpt["step"] == 42
    assert ckpt["val_loss"] == 1.23

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_load_missing_checkpoint_raises(tmp_path):
    _, model, _ = _tiny_setup()
    raised = False
    try:
        load_checkpoint(str(tmp_path / "nope.pt"), model)
    except FileNotFoundError:
        raised = True
    assert raised


def test_checkpoint_rotation_keeps_last_n(tmp_path):
    cfg, model, optimizer = _tiny_setup()
    ckpt_dir = tmp_path / "ckpts"

    for step in range(1, 6):
        save_checkpoint(
            model, optimizer, scaler=None, step=step, val_loss=1.0,
            model_cfg_dict={}, train_cfg_dict={}, ckpt_dir=str(ckpt_dir), tag="latest", keep_last_n=2,
        )

    step_ckpts = sorted((ckpt_dir).glob("ckpt_step*.pt"))
    assert len(step_ckpts) == 2
