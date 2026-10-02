"""CLI Training Script for CaptionCraft."""

import argparse
from src.captioncraft.config import CaptionCraftConfig
from src.captioncraft.trainer import CaptionTrainer
from src.captioncraft.dataset import Flickr8kDataset
from src.captioncraft.tokenizer import get_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train CaptionCraft Vision-Language Captioning Model")
    parser.add_argument("--data_dir", type=str, default="data/flickr-8k", help="Path to Flickr8k root directory")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--lr_cnn", type=float, default=1e-5, help="Learning rate for ConvNeXt backbone")
    parser.add_argument("--lr_transformer", type=float, default=1e-4, help="Learning rate for Transformer decoder")
    args = parser.parse_args()

    cfg = CaptionCraftConfig(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        cnn_lr=args.lr_cnn,
        transformer_lr=args.lr_transformer,
    )

    tokenizer = get_tokenizer()
    print("CaptionCraft Trainer initialized.")
    print(f"Configuration: {cfg.model_dump()}")

    trainer = CaptionTrainer(config=cfg, checkpoint_dir=args.checkpoint_dir)
    # If live training data and PyTorch are present, run training
    try:
        import torch
        from torch.utils.data import DataLoader
        from src.captioncraft.models import CaptionModel
        from src.captioncraft.transforms import get_transforms

        tfms = get_transforms(target_size=cfg.img_size)
        train_ds = Flickr8kDataset(cfg.data_dir, data_split="train", transform=tfms, tokenizer=tokenizer, phase="train")
        eval_ds = Flickr8kDataset(cfg.data_dir, data_split="dev", transform=tfms, tokenizer=tokenizer, phase="train")

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size)

        model = CaptionModel(
            emb_size=cfg.emb_dim,
            nhead=cfg.nhead,
            num_decoder_layers=cfg.num_layers,
            tgt_vocab_size=len(tokenizer),
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )
        trainer.train(model, train_loader, eval_loader, tokenizer)
    except Exception as e:
        print(f"Training halted: {e}")

if __name__ == "__main__":
    main()
