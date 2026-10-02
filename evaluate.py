"""CLI Evaluation Script for CaptionCraft Benchmarking."""

import argparse
from src.captioncraft.config import CaptionCraftConfig
from src.captioncraft.inference import CaptionPredictor
from src.captioncraft.dataset import Flickr8kDataset
from src.captioncraft.metrics import evaluate_corpus

def main():
    parser = argparse.ArgumentParser(description="Evaluate CaptionCraft Checkpoint on Test Set")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default="data/flickr-8k", help="Path to Flickr8k root directory")
    parser.add_argument("--split", type=str, default="test", help="Data split to evaluate on ('test' or 'dev')")
    args = parser.parse_args()

    print(f"Evaluating CaptionCraft on {args.split} split...")
    predictor = CaptionPredictor(checkpoint_path=args.checkpoint if args.checkpoint else None)
    test_ds = Flickr8kDataset(args.data_dir, data_split=args.split, phase="test")

    predictions = []
    references = []

    limit = min(50, len(test_ds))
    for i in range(limit):
        item = test_ds[i]
        res = predictor.predict(item["image"])
        predictions.append(res["caption"])
        references.append(item["caption"])

    metrics = evaluate_corpus(predictions, references)
    print("=" * 45)
    print("CaptionCraft Evaluation Results:")
    print(f"- BLEU-1:          {metrics['bleu_1']:.4f}")
    print(f"- BLEU-4:          {metrics['bleu_4']:.4f}")
    print(f"- ROUGE-L F1:      {metrics['rougeL_fmeasure']:.4f}")
    print("=" * 45)

if __name__ == "__main__":
    main()
