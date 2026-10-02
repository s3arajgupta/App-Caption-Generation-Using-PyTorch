"""Unit tests for tokenizer functionality."""

from src.captioncraft.tokenizer import SimpleFallbackTokenizer, get_tokenizer

def test_simple_fallback_tokenizer_special_tokens():
    tok = SimpleFallbackTokenizer()
    assert tok.bos_token == "<|startoftext|>"
    assert tok.eos_token == "<|endoftext|>"
    assert tok.pad_token == "[PAD]"
    assert tok.unk_token == "<|unk|>"

def test_simple_fallback_tokenizer_encode_decode():
    tok = SimpleFallbackTokenizer()
    text = "A dog is playing in the grass."
    encoded = tok.encode(text)
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    decoded = tok.decode(encoded)
    assert "dog" in decoded
    assert "grass" in decoded

def test_get_tokenizer_contract():
    tok = get_tokenizer()
    assert hasattr(tok, "encode")
    assert hasattr(tok, "decode")
    assert hasattr(tok, "bos_token")
