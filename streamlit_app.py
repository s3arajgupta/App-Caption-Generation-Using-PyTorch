"""CaptionCraft: Vision-Language Multimodal Image Captioning Dashboard.

Built with Streamlit, ConvNeXt-Small, and Autoregressive Transformer Decoder.
"""

import os
from pathlib import Path
from PIL import Image
import streamlit as st

from src.captioncraft.config import CaptionCraftConfig, get_default_config
from src.captioncraft.inference import CaptionPredictor
from src.captioncraft.metrics import compute_bleu, compute_rouge_l

# Streamlit Page Config
st.set_page_config(
    page_title="CaptionCraft — Vision-Language Captioning",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .caption-box {
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 1.2rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .token-chip {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 0.2rem 0.6rem;
        margin: 0.2rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# State initialization
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "image_name" not in st.session_state:
    st.session_state.image_name = ""

def load_preset(image_path: str, name: str):
    p = Path(image_path)
    if p.exists():
        st.session_state.selected_image = Image.open(p).convert("RGB")
        st.session_state.image_name = name

def main():
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Controls")
        
        cfg = get_default_config()
        
        st.subheader("Model Architecture")
        st.markdown("**Encoder:** `ConvNeXt-Small`")
        st.markdown("**Decoder:** `6-Layer Transformer`")
        st.markdown("**Tokenizer:** `GPT-2 BPE (50,260 tokens)`")
        
        st.markdown("---")
        st.subheader("Generation Settings")
        max_tokens = st.slider("Max Output Length", min_value=10, max_value=80, value=50)
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=1.0, step=0.1)

        st.markdown("---")
        st.subheader("⚡ Quick Preset Gallery")
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            if st.button("🐕 Dog", use_container_width=True):
                load_preset("assets/samples/dog_ball.png", "dog_ball.png")
        with col_g2:
            if st.button("🐱 Cat", use_container_width=True):
                load_preset("assets/samples/cat_window.png", "cat_window.png")
        with col_g3:
            if st.button("🏔️ Lake", use_container_width=True):
                load_preset("assets/samples/mountain_lake.png", "mountain_lake.png")

        st.markdown("---")
        st.subheader("📁 Upload Custom Image")
        uploaded_file = st.file_uploader(
            "Upload image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload an image to generate description.",
        )
        if uploaded_file is not None:
            st.session_state.selected_image = Image.open(uploaded_file).convert("RGB")
            st.session_state.image_name = uploaded_file.name

    # Header
    st.markdown('<div class="main-title">CaptionCraft 🎨</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Deep Multimodal Vision-Language Captioning with ConvNeXt-Small and Autoregressive Transformer Decoder</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Captioning Studio",
        "🏛️ Neural Architecture",
        "🧪 Evaluation Metric Lab",
        "📈 Training Hyperparameters",
    ])

    # Tab 1: Studio
    with tab1:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Input Visual")
            if st.session_state.selected_image is not None:
                st.image(st.session_state.selected_image, caption=f"Active: {st.session_state.image_name}", use_container_width=True)
            else:
                st.info("👈 Select a sample from the sidebar presets or upload an image to begin.")
                # Show thumbnails of presets
                p1, p2, p3 = st.columns(3)
                sample_p = Path("assets/samples")
                if (sample_p / "dog_ball.png").exists():
                    with p1:
                        st.image(str(sample_p / "dog_ball.png"), caption="Dog & Ball")
                if (sample_p / "cat_window.png").exists():
                    with p2:
                        st.image(str(sample_p / "cat_window.png"), caption="Cat in Sun")
                if (sample_p / "mountain_lake.png").exists():
                    with p3:
                        st.image(str(sample_p / "mountain_lake.png"), caption="Mountain Lake")

        with c2:
            st.subheader("Generated Caption & Intelligence")
            if st.session_state.selected_image is not None:
                predictor = CaptionPredictor()
                with st.spinner("Decoding visual representations..."):
                    result = predictor.predict(
                        st.session_state.selected_image,
                        max_length=max_tokens,
                        temperature=temperature,
                    )

                st.markdown(f'<div class="caption-box">“{result["caption"]}”</div>', unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Latency", f"{result['latency_ms']} ms")
                with m2:
                    st.metric("Generated Tokens", len(result["tokens"]))
                with m3:
                    st.metric("Confidence", f"{round(result['confidence'] * 100, 1)}%")

                st.markdown("---")
                st.subheader("Token Decomposition")
                token_html = "".join([f'<span class="token-chip">{tok}</span>' for tok in result["tokens"]])
                st.markdown(token_html, unsafe_allow_html=True)
            else:
                st.write("Awaiting image input...")

    # Tab 2: Architecture
    with tab2:
        st.subheader("Multimodal System Architecture")
        st.markdown("""
        CaptionCraft employs a modular **encoder-decoder vision-language framework** engineered for high-fidelity caption generation:
        
        1. **Vision Backbone (ConvNeXt-Small)**:
           - Ingests **224 × 224** resolution images with aspect-ratio-preserving symmetric padding.
           - Employs 7×7 depthwise convolutions and inverted bottleneck stages.
           - Outputs spatial feature maps of dimension `(Batch, 768, 7, 7)` which are flattened into **49 visual tokens** of 768-D.
           
        2. **Sinusoidal Positional Encoding**:
           - Computes multi-frequency sine and cosine spatial position representations across visual and text tokens.
           
        3. **Autoregressive Transformer Decoder**:
           - **6 stacked decoder layers** with **8-head Multi-Head Self-Attention** and **Cross-Attention**.
           - Look-ahead causal masking ensures autoregressive conditioning: $P(y_t | y_{<t}, I)$.
           - **GELU activation** with 2,048-dimensional feedforward hidden layers.
           
        4. **GPT-2 BPE Tokenizer**:
           - 50,260 subword tokens with dedicated `<|startoftext|>`, `<|endoftext|>`, and `[PAD]` boundaries.
        """)

    # Tab 3: Evaluation Metrics Lab
    with tab3:
        st.subheader("Automated Metric Benchmarking")
        st.markdown("Evaluate candidate captions against ground-truth references using **BLEU (1–4)** and **ROUGE-L**.")

        sample_hyp = st.text_input(
            "Hypothesis / Candidate Caption",
            value="A golden retriever dog catching a red ball on green grass.",
        )
        sample_refs = st.text_area(
            "Reference Captions (one per line)",
            value="A dog catches a red ball in the park.\nA golden dog playing with a red ball on the lawn.\nA dog leaping to catch a ball outside on the grass.",
            height=100,
        )

        refs_list = [r.strip() for r in sample_refs.split("\n") if r.strip()]

        if st.button("Calculate Metrics"):
            b_scores = compute_bleu(sample_hyp, refs_list)
            r_scores = compute_rouge_l(sample_hyp, refs_list)

            c_b1, c_b2, c_b3, c_b4, c_rl = st.columns(5)
            with c_b1:
                st.metric("BLEU-1", b_scores["bleu_1"])
            with c_b2:
                st.metric("BLEU-2", b_scores["bleu_2"])
            with c_b3:
                st.metric("BLEU-3", b_scores["bleu_3"])
            with c_b4:
                st.metric("BLEU-4", b_scores["bleu_4"])
            with c_rl:
                st.metric("ROUGE-L F1", r_scores["rougeL_fmeasure"])

    # Tab 4: Hyperparameters
    with tab4:
        st.subheader("Training Configurations & Dual-Optimizer Scheme")
        st.markdown("""
        To prevent catastrophic forgetting in the pretrained ConvNeXt backbone while allowing rapid convergence in the Transformer decoder, CaptionCraft uses **differential learning rates**:
        """)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("### 🖼️ Vision Encoder (ConvNeXt)")
            st.markdown("- **Optimizer:** `AdamW`")
            st.markdown("- **Learning Rate:** `1e-5` (conservative fine-tuning)")
            st.markdown("- **Scheduler:** `CosineAnnealingLR` (T_max=35)")
        with col_t2:
            st.markdown("### 📝 Text Decoder (Transformer)")
            st.markdown("- **Optimizer:** `AdamW`")
            st.markdown("- **Learning Rate:** `1e-4` (active generative training)")
            st.markdown("- **Scheduler:** `CosineAnnealingLR` (T_max=35)")

if __name__ == "__main__":
    main()
