import pytest

from transformer_project.config import ModelConfig, RunConfig, TrainConfig


def test_model_config_rejects_bad_head_split():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=8)  # 100 not divisible by 8


def test_model_config_rejects_bad_gqa_split():
    with pytest.raises(ValueError):
        ModelConfig(n_heads=8, n_kv_heads=3)  # 8 not divisible by 3


def test_train_config_rejects_bad_amp_dtype():
    with pytest.raises(ValueError):
        TrainConfig(amp_dtype="fp8")


def test_train_config_rejects_warmup_exceeding_max_steps():
    with pytest.raises(ValueError):
        TrainConfig(max_steps=100, warmup_steps=200)


def test_runconfig_roundtrip_yaml(tmp_path):
    cfg = RunConfig()
    cfg.model.d_model = 256
    cfg.train.lr = 1e-4
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)

    loaded = RunConfig.from_yaml(path)
    assert loaded.model.d_model == 256
    assert loaded.train.lr == 1e-4
