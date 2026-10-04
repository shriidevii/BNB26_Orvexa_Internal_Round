# TrustLayer
## Multimodal Digital Authenticity & Trust

> **Evidence, Not Just a Verdict.**

TrustLayer is an AI-powered multimodal digital forensics platform designed to detect manipulated, synthetic, and out-of-distribution digital content. 

Rather than treating media evaluation as a black-box binary classification, TrustLayer correlates visual artifacts, acoustic signatures, and cross-modal semantic consistency to deliver calibrated, verifiable forensic evidence.

---

### Key Capabilities
- **Decoupled Expert Processing:** Independent extraction of visual and acoustic representations.
- **Cross-Modal Semantic Alignment:** Detects dissonance between imagery, spoken audio, and transcripts (e.g., authentic video with synthetic audio overlay).
- **Inductive Conformal Prediction:** Statistically guaranteed uncertainty bounds ($\alpha = 0.10$, 90% coverage guarantee) to eliminate overconfident false positives.
- **Signal-Domain Forensics:** 2D-FFT frequency magnitude spectrum analysis and acoustic spectro-temporal density inspection.
- **Explainable Evidence Graph:** Dynamic causal graph generation tracing signals from raw evidence to final determination.
- **Cryptographic Provenance (C2PA-Compliant):** Automated generation of SHA-256 hashed JSON audit manifests.

---

## 1. Problem Statement

Generative AI tools make it trivial to produce photorealistic images, synthetic voices, and synchronized multimedia manipulations. Current deepfake detection tools suffer from three fundamental limitations:

1. **Unimodal Blindspots:** They evaluate video or audio in isolation, failing when authentic footage is paired with cloned audio.
2. **Overconfident Black Boxes:** Standard models force high-confidence predictions on heavily compressed or degraded media, driving high false-positive rates.
3. **Lack of Forensic Chain of Custody:** Outputting an isolated percentage score fails to provide admissible audit trails for legal, intelligence, or newsroom workflows.

TrustLayer addresses these limitations through synchronized cross-modal verification.

---

## 2. Solution Overview

TrustLayer operates a multi-stage forensic pipeline across separate sensory domains:

### Image Expert
* **Backbone:** `EfficientNet-B0`
* **Function:** Extracts high-dimensional visual feature representations to identify spatial artifacts, boundary blending errors, and generative lattice anomalies.

### Audio Expert
* **Backbone:** `Wav2Vec 2.0`
* **Function:** Analyzes raw waveforms to isolate acoustic representations associated with neural vocoders, phase degradation, and unnatural formants.

### Multimodal Fusion Engine
* **Architecture:** Multi-Layer Perceptron (MLP)
* **Function:** Concatenates visual and acoustic embedding vectors to evaluate holistic asset integrity rather than relying on a single point of failure.

### Cross-Modal Semantic Analysis
* **Function:** Compares visual contextual descriptors against speech transcripts to detect content mismatch, contextual discrepancies, and out-of-sync audio-visual pairing.

### Conformal Uncertainty Calibration
* **Algorithm:** Inductive Conformal Prediction (ICP)
* **Function:** Applies an empirically calibrated error margin ($\hat{q} = 0.310$) derived from held-out calibration sets. Ambiguous, degraded samples are automatically routed to an `INCONCLUSIVE` status for human verification rather than risking false classification.

---

## 3. Core Architecture

```text
                     ┌─────────────────────┐
                     │   Image + Audio     │
                     │       Input         │
                     └──────────┬──────────┘
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
        ┌──────────────────┐        ┌──────────────────┐
        │   Image Expert   │        │   Audio Expert   │
        │  EfficientNet-B0 │        │     Wav2Vec2     │
        └────────┬─────────┘        └────────┬─────────┘
                 │                           │
                 │ Image Features            │ Audio Features
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                     ┌──────────────────┐
                     │ Multimodal Fusion│
                     │       MLP        │
                     └─────────┬────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ Cross-Modal Check│
                     │ (Semantic Disson)│
                     └─────────┬────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │Conformal Boundary│
                     │ (Margin q = 0.31)│
                     └─────────┬────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Evidence Graph  │         │  C2PA Manifest  │
        │ (Explainability)│         │ (SHA-256 Audit) │
        └─────────────────┘         └─────────────────┘

4. Dataset StrategyTrustLayer uses a hybrid benchmark strategy pairing verified organic media against controlled synthetic manipulations:Real Visual Baseline: Curated splits from the Common Objects in Context (COCO) dataset.Real Acoustic Baseline: Uncompressed speech recordings and transcripts from LibriSpeech.Perturbation & Degradation Suite: Benchmarks include realistic social media degradation (JPEG compression, Gaussian noise, vocoder downsampling) to evaluate real-world resilience.5. Model Training & Pipeline StagesTrustLayer uses transfer learning with fine-tuned forensic heads:Visual Pipeline: Input Image $\rightarrow$ EfficientNet-B0 $\rightarrow$ Spatial Feature Embeddings $\rightarrow$ Binary Classification Head.Audio Pipeline: Raw Waveform $\rightarrow$ Wav2Vec2 $\rightarrow$ Acoustic Temporal Embeddings $\rightarrow$ Classification Head.Fusion Pipeline: Concat(Image Embeddings, Audio Embeddings) $\rightarrow$ Fusion MLP $\rightarrow$ Multimodal Risk Score.Calibration Stage: Non-conformity scoring on held-out splits to establish the conformal quantile $\hat{q}$.6. Physical Signals & Graph ExplainabilityTo provide defensible forensic analysis, TrustLayer computes signal-domain diagnostics:2D-FFT Frequency Magnitude Spectrum: Identifies high-frequency checkerboard artifacts characteristic of GAN and diffusion generation.Acoustic Spectro-Temporal Energy Density: Uncovers vocoder harmonic dropouts and speech synthesis anomalies.Directed Evidence Graph: Uses NetworkX to map inter-modality dependencies and relationship weights leading to the verdict.7. Technology StackCore Runtime: Python 3.10+Deep Learning Frameworks: PyTorch, TorchVision, TorchAudio, Transformers, Scikit-learnFeature Extraction & Signals: NumPy, SciPy, Pillow, Librosa, SoundFileVisual Analytics & Graph Theory: Matplotlib, Seaborn, NetworkXInterface & Serving: Streamlit8. Repository StructurePlaintextBNB26_Orvexa_Internal_Round/
│
├── .streamlit/
│   └── config.toml                  # Global application theme
│
├── app/
│   ├── app.py                       # Application landing page
│   ├── core/
│   │   ├── pipeline.py              # Multimodal inference pipeline
│   │   ├── spectral_inspector.py    # 2D-FFT & spectrogram generators
│   │   └── custody_exporter.py      # C2PA JSON manifest exporter
│   └── pages/
│       ├── 1_Investigate.py         # Forensic analysis workspace
│       ├── 2_Evidence_Graph.py      # Causal relationship mapping
│       ├── 3_Evaluation.py          # Benchmark metrics & ROC/AUC curves
│       ├── 4_Generalization.py      # Out-of-distribution evaluation
│       └── 5_Methodology.py         # Mathematical proofs & pipeline documentation
│
├── data/
│   ├── cases.csv                    # Ground-truth evaluation registry
│   ├── generated/                   # Synthetic benchmark artifacts
│   └── raw/                         # Pristine baseline media
│
├── models/                          # Serialized weights (.pt, .pth)
│
├── scripts/
│   ├── setup_data.py                # Dataset initialization & structure setup
│   ├── gen_fakes.py                 # Synthetic generation pipeline
│   ├── patch_audio.py               # Audio degradation & synthesis
│   └── scale_cases.py               # Dataset combinatorial scaling
│
├── training/
│   ├── 01_extract_image_embeddings.py
│   ├── 02_extract_audio_embeddings.py
│   ├── 03_train_experts.py
│   ├── 04_extract_consistency_features.py
│   ├── 05_train_fusion.py
│   └── 06_calibrate_conformal.py
│
├── requirements.txt
└── README.md
9. Installation & QuickstartStep 1: Clone RepositoryBashgit clone [https://github.com/shridevi/BNB26_Orvexa_Internal_Round.git](https://github.com/shridevi/BNB26_Orvexa_Internal_Round.git)
cd BNB26_Orvexa_Internal_Round
Step 2: Environment SetupPowerShellpython -m venv .venv
.venv\Scripts\Activate.ps1
Step 3: Install DependenciesPowerShellpip install -r requirements.txt
Step 4: Dataset InitializationPowerShellpython scripts/setup_data.py
Step 5: Launch DashboardPowerShellstreamlit run app/app.py
10. Training WorkflowTo retrain the experts or recalibrate the conformal thresholds:PowerShell# Extract modality representations
python training/01_extract_image_embeddings.py
python training/02_extract_audio_embeddings.py

# Train individual experts & extract consistency
python training/03_train_experts.py
python training/04_extract_consistency_features.py

# Train fusion layer & compute conformal boundary
python training/05_train_fusion.py
python training/06_calibrate_conformal.py
11. Core DifferentiatorStandard Deepfake DetectorsTrustLayer Forensic EngineUnimodal (Image or Audio only)Synchronized Multimodal Cross-ValidationBinary "Black Box" PredictionExplainable Evidence Graph & Spectral DiagnosticsForced outputs on degraded mediaInductive Conformal Prediction ($\pm 15.5\%$ Safety Boundary)Non-reproducible web outputsSHA-256 Hashed C2PA-Compliant JSON Audit Manifests
