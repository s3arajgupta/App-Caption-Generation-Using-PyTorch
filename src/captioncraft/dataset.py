"""Dataset and DataLoader utilities for Flickr8k image-caption pairs."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        pass

class Flickr8kDataset(Dataset):
    """Dataset loader for Flickr8k image and text directory format."""
    def __init__(
        self,
        root_dir: str,
        data_split: str = "train",
        transform: Any = None,
        tokenizer: Any = None,
        max_len: int = 50,
        phase: str = "train",
    ):
        self.root_dir = Path(root_dir)
        self.img_folder = self.root_dir / "images"
        self.text_folder = self.root_dir / "text"
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.phase = phase

        self.samples: List[Dict[str, Any]] = []
        split_file = self.text_folder / f"Flickr_8k.{data_split}Images.txt"
        token_file = self.text_folder / "Flickr8k.token.txt"

        if split_file.exists() and token_file.exists():
            self._load_dataset(split_file, token_file)
        else:
            # Fallback sample dataset if local flickr-8k directory not present
            self._load_fallback_samples()

    def _load_dataset(self, split_file: Path, token_file: Path):
        valid_images = set(line.strip() for line in open(split_file, "r", encoding="utf-8") if line.strip())
        captions_map: Dict[str, List[str]] = {}

        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    img_id = parts[0].split("#")[0]
                    cap = parts[1]
                    if img_id in valid_images:
                        if img_id not in captions_map:
                            captions_map[img_id] = []
                        captions_map[img_id].append(cap)

        for img_name, caps in captions_map.items():
            if self.phase == "train":
                for c in caps:
                    self.samples.append({"image_name": img_name, "caption": c})
            else:
                self.samples.append({"image_name": img_name, "caption": caps})

    def _load_fallback_samples(self):
        """Provide fallback samples for testing and demonstration."""
        fallback = [
            ("dog_ball.png", "A golden retriever dog catching a red ball on green grass."),
            ("cat_window.png", "A dark cat sitting by the window looking outside in the sun."),
            ("mountain_lake.png", "A scenic mountain range reflecting in a calm blue lake at sunset.")
        ]
        for img, cap in fallback:
            if self.phase == "train":
                self.samples.append({"image_name": img, "caption": cap})
            else:
                self.samples.append({"image_name": img, "caption": [cap]})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_name = item["image_name"]
        
        # Look in img_folder or assets/samples
        img_path = self.img_folder / image_name
        if not img_path.exists():
            local_fallback = Path("assets/samples") / image_name
            if local_fallback.exists():
                img_path = local_fallback

        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform is not None:
            image = self.transform(image)

        caption_data = item["caption"]
        if self.phase == "train" and self.tokenizer is not None:
            raw_cap = f"{self.tokenizer.bos_token} {caption_data} {self.tokenizer.eos_token}"
            encoded = self.tokenizer.encode(raw_cap)
            pad_count = max(0, self.max_len - len(encoded))
            padded_tokens = encoded + [self.tokenizer.pad_token_id] * pad_count
            padded_tokens = padded_tokens[:self.max_len]
            if HAS_TORCH:
                caption_tensor = torch.tensor(padded_tokens, dtype=torch.long)
            else:
                caption_tensor = padded_tokens
            return {"image": image, "caption": caption_tensor, "image_name": image_name}
        
        return {"image": image, "caption": caption_data, "image_name": image_name}


def eval_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate single evaluation item preserving list of reference captions."""
    first = batch[0]
    img = first["image"]
    if HAS_TORCH and hasattr(img, "unsqueeze"):
        img = img.unsqueeze(0)
    return {
        "image": img,
        "caption": first["caption"],
        "image_name": first["image_name"]
    }
