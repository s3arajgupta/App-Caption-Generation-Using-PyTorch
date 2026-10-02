"""Neural Network Architectures for CaptionCraft: ConvNeXt + Transformer Decoder."""

import math
from typing import Optional, Any

try:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision.models import convnext_small, ConvNeXt_Small_Weights
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Fallback placeholder base class
    class nn:
        class Module:
            pass

if HAS_TORCH:
    class PositionalEncoding(nn.Module):
        """Sinusoidal Positional Encoding for token sequences and spatial image tokens."""
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Args: x shape [seq_len, batch_size, embedding_dim]"""
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)

    def get_cnn_encoder(pretrained: bool = True) -> nn.Module:
        """Instantiate ConvNeXt-Small backbone stripped of classification head."""
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_small(weights=weights)
        # Strip pooling and classifier head to output feature map (B, 768, 7, 7)
        backbone = nn.Sequential(*list(model.children())[:-2])
        return backbone

    class TransformerDecoder(nn.Module):
        """Autoregressive Transformer Decoder for text generation conditioned on visual tokens."""
        def __init__(
            self,
            emb_size: int = 768,
            nhead: int = 8,
            num_decoder_layers: int = 6,
            tgt_vocab_size: int = 50260,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "gelu",
        ):
            super().__init__()
            self.emb_size = emb_size

            self.embedding = nn.Embedding(tgt_vocab_size, emb_size)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=emb_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            self.text_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
            self.generator = nn.Linear(emb_size, tgt_vocab_size)
            self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

            self.init_weights()

        def init_weights(self):
            range_val = 0.1
            self.embedding.weight.data.uniform_(-range_val, range_val)
            self.generator.bias.data.zero_()
            self.generator.weight.data.uniform_(-range_val, range_val)

        def forward(
            self,
            src_emb: torch.Tensor,
            tgt_tokens: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            tgt_padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward pass during training with teacher forcing and causal masking."""
            B, D, H, W = src_emb.shape
            # Flatten spatial grid into sequence: [B, D, H*W] -> [H*W, B, D]
            src_emb = src_emb.reshape(B, D, -1).permute(2, 0, 1)
            src_emb = self.positional_encoding(src_emb)

            tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.emb_size)
            tgt_emb = tgt_emb.permute(1, 0, 2)
            tgt_emb = self.positional_encoding(tgt_emb)

            outs = self.text_decoder(
                tgt_emb, src_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask
            )
            return self.generator(outs)

        def generate(self, img_ft: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
            """Step-by-step autoregressive token prediction."""
            src_emb = self.positional_encoding(img_ft)
            tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.emb_size)
            tgt_emb = tgt_emb.permute(1, 0, 2)
            tgt_emb = self.positional_encoding(tgt_emb)

            outs = self.text_decoder(tgt_emb, src_emb)
            return self.generator(outs)

    class CaptionModel(nn.Module):
        """End-to-End Multimodal Vision-Language Captioning System."""
        def __init__(
            self,
            emb_size: int = 768,
            nhead: int = 8,
            num_decoder_layers: int = 6,
            tgt_vocab_size: int = 50260,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "gelu",
            pretrained_cnn: bool = True,
        ):
            super().__init__()
            self.image_encoder = get_cnn_encoder(pretrained=pretrained_cnn)
            self.text_decoder = TransformerDecoder(
                emb_size=emb_size,
                nhead=nhead,
                num_decoder_layers=num_decoder_layers,
                tgt_vocab_size=tgt_vocab_size,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )

        def forward(
            self,
            imgs: torch.Tensor,
            tgt_tokens: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            tgt_padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            src_emb = self.image_encoder(imgs)
            return self.text_decoder(src_emb, tgt_tokens, tgt_mask, tgt_padding_mask)

    def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
        """Generate upper-triangular causal attention mask to prevent attending to future tokens."""
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int, device: str = "cpu"):
        """Create causal look-ahead mask and key-padding mask for target sequence."""
        tgt_seq_len = tgt.shape[1]
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)
        tgt_padding_mask = (tgt == pad_idx)
        return tgt_mask, tgt_padding_mask

else:
    # Stub classes for test/inspection when torch is absent
    class PositionalEncoding:
        pass
    def get_cnn_encoder(pretrained: bool = True):
        return None
    class TransformerDecoder:
        pass
    class CaptionModel:
        pass
    def generate_square_subsequent_mask(sz: int, device: str = "cpu"):
        return None
    def create_mask(src, tgt, pad_idx, device="cpu"):
        return None, None
