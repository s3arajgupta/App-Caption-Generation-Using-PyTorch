"""Configuration module for CaptionCraft vision-language captioning system."""

import os
from typing import List, Tuple
from pydantic import BaseModel, Field

class CaptionCraftConfig(BaseModel):
    """Configuration hyperparameters and environment paths."""
    
    # Model architecture parameters
    img_size: int = Field(default=224, description="Input image resolution (height, width)")
    emb_dim: int = Field(default=768, description="Latent embedding dimension (ConvNeXt & Transformer)")
    nhead: int = Field(default=8, description="Number of multi-head attention heads")
    num_layers: int = Field(default=6, description="Number of Transformer Decoder layers")
    dim_feedforward: int = Field(default=2048, description="Transformer feedforward expansion dimension")
    dropout: float = Field(default=0.1, description="Dropout probability")
    activation: str = Field(default="gelu", description="Decoder activation function")
    
    # Vocabulary & sequence constraints
    max_len: int = Field(default=50, description="Maximum generated token sequence length")
    vocab_size: int = Field(default=50260, description="Tokenizer vocabulary size including special tokens")
    bos_token: str = Field(default="<|startoftext|>", description="Beginning-of-sequence token")
    eos_token: str = Field(default="<|endoftext|>", description="End-of-sequence token")
    pad_token: str = Field(default="[PAD]", description="Padding token")
    unk_token: str = Field(default="<|unk|>", description="Unknown token")
    
    # Image normalization (ImageNet standard)
    mean: Tuple[float, float, float] = Field(default=(0.485, 0.456, 0.406))
    std: Tuple[float, float, float] = Field(default=(0.229, 0.224, 0.225))
    
    # Training & optimization
    batch_size: int = Field(default=8, description="Batch size for training and validation")
    num_epochs: int = Field(default=35, description="Number of total training epochs")
    cnn_lr: float = Field(default=1e-5, description="Learning rate for ConvNeXt backbone")
    transformer_lr: float = Field(default=1e-4, description="Learning rate for Transformer decoder")
    seed: int = Field(default=42, description="Global random seed")
    
    # Storage & directories
    data_dir: str = Field(default="data/flickr-8k", description="Flickr8k dataset path")
    checkpoint_dir: str = Field(default="checkpoints", description="Model checkpoint export path")
    device: str = Field(default="cuda", description="Compute device ('cuda' or 'cpu')")

    def model_post_init(self, __context):
        # Auto-detect CUDA availability if PyTorch is installed
        try:
            import torch
            if not torch.cuda.is_available() and self.device == "cuda":
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

def get_default_config() -> CaptionCraftConfig:
    """Instantiate standard production config."""
    return CaptionCraftConfig()
