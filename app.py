"""Gradio Web Application for CaptionCraft Image Captioning."""

import os
from pathlib import Path
from PIL import Image

from src.captioncraft.inference import CaptionPredictor

predictor = CaptionPredictor()

def predict_caption(img: Image.Image) -> str:
    """Predict caption using CaptionCraft."""
    if img is None:
        return "Please upload or select an image."
    res = predictor.predict(img)
    return res["caption"]

def main():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Run 'pip install gradio' or use 'streamlit run streamlit_app.py'.")
        return

    sample_dir = Path("assets/samples")
    sample_images = []
    if sample_dir.exists():
        for p in sample_dir.glob("*.png"):
            sample_images.append([str(p)])

    demo = gr.Interface(
        fn=predict_caption,
        inputs=gr.Image(type="pil", label="Upload an Image"),
        outputs=gr.Textbox(label="Generated Caption", lines=2),
        examples=sample_images if sample_images else None,
        title="CaptionCraft 🎨 — Vision-Language Image Captioning",
        description="Deep learning multimodal captioning powered by ConvNeXt-Small and Autoregressive Transformer Decoder with GPT-2 BPE.",
    )
    demo.launch(share=False)

if __name__ == "__main__":
    main()
