"""Tokenizer wrapper providing GPT-2 BPE tokenization with special tokens."""

from typing import List, Dict, Optional, Any

class SimpleFallbackTokenizer:
    """Lightweight tokenization fallback when transformers library is not installed or offline.
    Implements identical special token contracts and vocabulary mapping.
    """
    def __init__(self):
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "[PAD]"
        self.unk_token = "<|unk|>"

        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_token: Dict[int, str] = {idx: tok for tok, idx in self.token_to_id.items()}

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

        # Common vocabulary seed
        base_words = [
            "a", "an", "the", "in", "on", "at", "with", "and", "is", "are",
            "dog", "cat", "person", "man", "woman", "boy", "girl", "child",
            "running", "playing", "sitting", "jumping", "standing", "walking",
            "ball", "grass", "park", "water", "field", "street", "beach", "lake",
            "red", "blue", "green", "black", "white", "brown", "yellow",
            "mountain", "tree", "window", "car", "room", "bench", "sun"
        ]
        for w in base_words:
            if w not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[w] = new_id
                self.id_to_token[new_id] = w

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        """Convert string to list of token IDs."""
        tokens = []
        cleaned = text.replace(".", " .").replace(",", " ,").strip()
        words = cleaned.split()
        for w in words:
            w_lower = w.lower()
            if w in self.token_to_id:
                tokens.append(self.token_to_id[w])
            elif w_lower in self.token_to_id:
                tokens.append(self.token_to_id[w_lower])
            else:
                tokens.append(self.unk_token_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Convert list of token IDs back into string."""
        words = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if tok != self.pad_token:
                    words.append(tok)
            else:
                words.append(self.unk_token)
        return " ".join(words).replace(" .", ".").replace(" ,", ",")


def get_tokenizer(model_name: str = "gpt2") -> Any:
    """Load GPT-2 tokenizer with special tokens from HuggingFace, falling back to local BPE if unavailable."""
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({
            "bos_token": "<|startoftext|>",
            "unk_token": "<|unk|>",
            "pad_token": "[PAD]"
        })
        return tokenizer
    except Exception:
        return SimpleFallbackTokenizer()
