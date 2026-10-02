"""Unit tests for inference and prediction pipeline."""

from PIL import Image
from src.captioncraft.inference import CaptionPredictor

def test_caption_predictor_pil():
    predictor = CaptionPredictor()
    img = Image.new("RGB", (300, 300), color=(100, 200, 100))
    res = predictor.predict(img)

    assert "caption" in res
    assert "tokens" in res
    assert "confidence" in res
    assert "latency_ms" in res
    assert len(res["caption"]) > 0

def test_caption_predictor_sample_files():
    predictor = CaptionPredictor()
    res = predictor.predict("assets/samples/dog_ball.png")
    assert "dog" in res["caption"].lower() or "ball" in res["caption"].lower()

    res_cat = predictor.predict("assets/samples/cat_window.png")
    assert "cat" in res_cat["caption"].lower() or "window" in res_cat["caption"].lower()
