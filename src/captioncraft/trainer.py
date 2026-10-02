"""Training and validation orchestration for CaptionCraft."""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from .config import CaptionCraftConfig, get_default_config

class CaptionTrainer:
    """Trainer orchestrator managing optimization, checkpointing, and scheduler stepping."""
    def __init__(
        self,
        config: Optional[CaptionCraftConfig] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        self.config = config or get_default_config()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, model: Any, train_loader: Any, eval_loader: Any, tokenizer: Any):
        """Execute complete multi-epoch training process with dual AdamW optimizers."""
        try:
            import torch
            from torch.nn import CrossEntropyLoss
            from torch.optim.lr_scheduler import CosineAnnealingLR
            from .models import create_mask
        except ImportError:
            print("PyTorch is required to execute live training.")
            return

        device = self.config.device
        model.to(device)

        criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        optimizer_cnn = torch.optim.AdamW(model.image_encoder.parameters(), lr=self.config.cnn_lr)
        optimizer_transformer = torch.optim.AdamW(model.text_decoder.parameters(), lr=self.config.transformer_lr)

        scheduler_cnn = CosineAnnealingLR(optimizer_cnn, T_max=self.config.num_epochs)
        scheduler_transformer = CosineAnnealingLR(optimizer_transformer, T_max=self.config.num_epochs)

        print(f"Starting training for {self.config.num_epochs} epochs on {device}...")

        for epoch in range(self.config.num_epochs):
            model.train()
            running_loss = 0.0
            steps = 0

            for data in train_loader:
                imgs = data["image"].to(device)
                target = data["caption"].to(device)

                target_in = target[:, :-1]
                target_out = target[:, 1:]

                model.zero_grad(set_to_none=True)
                tgt_mask, tgt_padding_mask = create_mask(
                    imgs, target_in, pad_idx=tokenizer.pad_token_id, device=device
                )

                logits = model(imgs, target_in, tgt_mask.to(device), tgt_padding_mask.to(device))
                T, B, D = logits.shape
                loss = criterion(logits.permute(1, 0, 2).reshape(T * B, D), target_out.reshape(T * B))

                loss.backward()
                optimizer_cnn.step()
                optimizer_transformer.step()

                running_loss += loss.item()
                steps += 1

            scheduler_cnn.step()
            scheduler_transformer.step()
            epoch_loss = running_loss / max(1, steps)
            print(f"Epoch [{epoch + 1}/{self.config.num_epochs}] Train Loss: {epoch_loss:.4f}")

            # Checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                ckpt_path = self.checkpoint_dir / f"model_epoch_{epoch + 1}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
