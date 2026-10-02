"""Unit tests for model architecture components and mask generation."""

from src.captioncraft.models import (
    HAS_TORCH,
    generate_square_subsequent_mask,
    create_mask,
    CaptionModel,
)

def test_model_interface_exists():
    assert CaptionModel is not None

def test_torch_conditional():
    if HAS_TORCH:
        import torch
        mask = generate_square_subsequent_mask(5)
        assert mask.shape == (5, 5)
        # Check causal property: diagonal and lower should be 0, upper should be -inf
        assert mask[0, 0] == 0.0
        assert mask[0, 1] == float('-inf')
    else:
        # Stub check
        assert generate_square_subsequent_mask(5) is None
