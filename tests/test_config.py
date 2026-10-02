"""Unit tests for CaptionCraft configuration."""

from src.captioncraft.config import CaptionCraftConfig, get_default_config

def test_default_config():
    cfg = get_default_config()
    assert cfg.img_size == 224
    assert cfg.emb_dim == 768
    assert cfg.nhead == 8
    assert cfg.num_layers == 6
    assert cfg.dim_feedforward == 2048
    assert cfg.dropout == 0.1
    assert cfg.activation == "gelu"
    assert cfg.max_len == 50
    assert cfg.batch_size == 8
    assert cfg.cnn_lr == 1e-5
    assert cfg.transformer_lr == 1e-4

def test_custom_config():
    cfg = CaptionCraftConfig(
        img_size=256,
        emb_dim=512,
        num_layers=4,
        batch_size=16
    )
    assert cfg.img_size == 256
    assert cfg.emb_dim == 512
    assert cfg.num_layers == 4
    assert cfg.batch_size == 16
