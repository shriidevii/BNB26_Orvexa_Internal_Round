import streamlit as st

st.set_page_config(page_title="Methodology | TrustLayer", page_icon="📚", layout="wide")
st.title("📚 Methodology & System Architecture")

st.markdown("""
## The Problem with Single-Modality Detection
Generative AI has evolved to the point where single modalities (an image, a voice clip) can easily bypass unimodal perceptual detectors. However, coordinated manipulation almost always leaves **cross-modal dissonance**. 

TrustLayer solves this by analyzing multiple artifacts simultaneously and mathematically checking for contradictions.

---

### 1. Feature-Store Accelerated Training
Instead of running heavy backbones end-to-end, we freeze foundation feature extractors:
* **Visual:** `EfficientNet-B0` (1280-dim penultimate representations)
* **Audio:** `Wav2Vec 2.0` (768-dim acoustic representations)
* **Cross-Modal:** `CLIP ViT` (Cosine similarity between image, caption, and spoken transcript)

### 2. Leakage-Safe Data Engineering
To prevent data leakage, we separate our training pools:
* **Pool E (Experts):** Real COCO val2017 and LibriSpeech data, plus controlled single-modality generations (SD-Turbo, edge-TTS). Used *only* to train unimodal expert heads.
* **Pool C (Multimodal Cases):** Paired test benches isolated by speaker and prompt to guarantee zero target leakage when training the fusion MLP.

### 3. Conformal Prediction (Uncertainty)
We do not use naive binary cutoffs (e.g., `if score > 0.5`). We use **Inductive Conformal Prediction** calibrated at a significance level of $\\alpha = 0.10$. 
If the model's non-conformity score spans multiple classes under distributional shift, the sample is flagged as **INCONCLUSIVE**, providing a mathematical safety guarantee against false accusations.

### 4. Zero-Shot Generalization
To prove the model isn't just memorizing one generator's signature, we train entirely on `SD-Turbo` and `edge-TTS`, but evaluate on held-out architectures like `SDXL-Turbo` (different latent scale) and `XTTS-v2` (autoregressive).
""")