"""Unit tests for image transformation and aspect-ratio padding."""

from PIL import Image
import numpy as np
from src.captioncraft.transforms import ResizePadTransform, preprocess_image_to_numpy

def test_resize_pad_landscape():
    # 400x200 landscape image
    img = Image.new("RGB", (400, 200), color=(255, 0, 0))
    transformer = ResizePadTransform(target_size=224)
    res = transformer(img)
    assert res.size == (224, 224)

def test_resize_pad_portrait():
    # 200x400 portrait image
    img = Image.new("RGB", (200, 400), color=(0, 255, 0))
    transformer = ResizePadTransform(target_size=224)
    res = transformer(img)
    assert res.size == (224, 224)

def test_resize_pad_square():
    # 300x300 square image
    img = Image.new("RGB", (300, 300), color=(0, 0, 255))
    transformer = ResizePadTransform(target_size=224)
    res = transformer(img)
    assert res.size == (224, 224)

def test_preprocess_image_to_numpy():
    img = Image.new("RGB", (350, 250), color=(100, 150, 200))
    arr = preprocess_image_to_numpy(img, target_size=224)
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32
