"""Unit tests for dataset loading and collation."""

from src.captioncraft.dataset import Flickr8kDataset, eval_collate_fn
from src.captioncraft.tokenizer import SimpleFallbackTokenizer
from src.captioncraft.transforms import ResizePadTransform

def test_dataset_fallback_loading():
    tok = SimpleFallbackTokenizer()
    tfm = ResizePadTransform(224)
    ds = Flickr8kDataset("non_existent_dir", data_split="train", transform=tfm, tokenizer=tok, phase="train")
    assert len(ds) >= 3

    item = ds[0]
    assert "image" in item
    assert "caption" in item
    assert "image_name" in item

def test_dataset_test_phase():
    ds = Flickr8kDataset("non_existent_dir", data_split="test", phase="test")
    item = ds[0]
    # In test phase, caption is list of reference strings
    assert isinstance(item["caption"], list)

def test_eval_collate_fn():
    batch = [{"image": "dummy_img", "caption": ["A test caption"], "image_name": "test.png"}]
    collated = eval_collate_fn(batch)
    assert collated["image_name"] == "test.png"
    assert collated["caption"] == ["A test caption"]
