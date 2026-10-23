# Deep Learning-Based Multimodal Emotion Detection: Fusion of Facial, Speech, and Text Modalities for Improved Accuracy

**Author:** Sai Bhadre (Roll No: SYMCA51 / PRN: MC25OB1069)  
**Guide:** Prof. Vaishali Hatkar (Assistant Professor)  
**Department:** Department of MCA  
**Institution:** Indira College of Engineering & Management, Pune (Autonomous, SPPU)  

---

## 📌 1. Project Overview & Abstract
Traditional emotion recognition systems rely heavily on a single modality (such as facial expressions, speech acoustics, or text sentiment), which often fails in real-world scenarios due to noise, ambiguous expressions, or lack of tonal context. 

This research project introduces an end-to-end **Deep Learning-Based Multimodal Emotion Detection Framework** that harmoniously fuses:
1. **Facial Modality ($X_f \to F_f$):** Visual expression cues extracted via Deep Residual CNN.
2. **Speech Modality ($X_s \to F_s$):** Acoustic & prosodic cues extracted via 1D Temporal CNN + Bidirectional LSTM + Attention Pooling.
3. **Text Modality ($X_t \to F_t$):** Contextual semantic representations extracted via Multi-Head Transformer Language Encoder.

The extracted representations are mapped into a shared multimodal space via projection layers and synthesized using **Cross-Modal Multi-Head Attention** and **Dynamic Gating Hybrid Fusion**. The system achieves higher accuracy and robustness compared to unimodal baselines across 7 emotional states: **Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral**.

---

## 🔬 2. Research Questions & Hypothesis

- **RQ1:** Does fusing facial, speech, and text modalities using deep learning improve emotion detection accuracy compared to unimodal models?
- **RQ2:** Which fusion strategy (early concatenation vs. cross-modal attention vs. hybrid gating) yields superior performance?
- **RQ3:** How robust is the proposed multimodal system when one or more modalities are noisy, degraded, or missing?
- **Hypothesis:** A deep learning-based multimodal fusion model will achieve significantly higher classification accuracy and Macro-F1 than any single-modality baseline.

---

## 📐 3. Mathematical Formulations & System Architecture

### 3.1 Modality Encoding
$$\begin{aligned}
F_f &= E_f(X_f) \in \mathbb{R}^{512} \quad &\text{(Facial Encoder)} \\
F_s &= E_s(X_s) \in \mathbb{R}^{256} \quad &\text{(Speech Encoder)} \\
F_t &= E_t(X_t) \in \mathbb{R}^{768} \quad &\text{(Text Encoder)}
\end{aligned}$$

### 3.2 Feature Projection (Common Multimodal Space)
Since encoders produce features of varying dimensions, linear projection layers with layer normalization align them into dimension $d_m = 256$:
$$\begin{aligned}
Z_f &= W_f F_f + b_f \in \mathbb{R}^{d_m} \\
Z_s &= W_s F_s + b_s \in \mathbb{R}^{d_m} \\
Z_t &= W_t F_t + b_t \in \mathbb{R}^{d_m}
\end{aligned}$$

### 3.3 Fusion Mechanisms
- **Early Concatenation Baseline:**
  $$Z_{\text{concat}} = [Z_f; Z_s; Z_t] \in \mathbb{R}^{3 \times d_m}$$
- **Cross-Modal Attention (Pairwise Interaction):**
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Hybrid Gating & Classifier:**
  $$\alpha_f, \alpha_s, \alpha_t = \text{Softmax}\left(W_g [Z_f; Z_s; Z_t]\right)$$
  $$Z_{\text{fusion}} = \text{MLP}\left([\alpha_f Z_f; \alpha_s Z_s; \alpha_t Z_t]\right)$$
  $$\hat{y} = \text{Softmax}(W_c Z_{\text{fusion}} + b_c)$$

---

## 📊 4. Experimental Configurations (Table I in Paper)

| ID | Configuration | Modalities | Description |
|---|---|---|---|
| **E1** | Facial only | Face ($X_f$) | Unimodal Visual Baseline |
| **E2** | Speech only | Speech ($X_s$) | Unimodal Acoustic Baseline |
| **E3** | Text only | Text ($X_t$) | Unimodal Textual Baseline |
| **E4** | Facial + Speech | Face + Speech | Bimodal Audio-Visual Fusion |
| **E5** | Speech + Text | Speech + Text | Bimodal Speech-Text Fusion |
| **E6** | Facial + Text | Face + Text | Bimodal Visual-Text Fusion |
| **E7** | Facial + Speech + Text | Face + Speech + Text | **Proposed Tri-modal Hybrid Fusion** |

**Evaluation Metrics:** Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), Macro-F1, and Confusion Matrix.

---

## 🗂️ 5. Project Directory Structure

```text
multimodal_emotion_detection/
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Dependency list
├── config/
│   └── config.py                      # Model hyperparameters, emotion classes, dims
├── data/
│   ├── raw/                           # Raw benchmark datasets (MELD, IEMOCAP)
│   ├── processed/                     # Preprocessed features & aligned data
│   ├── sample_inputs/                 # Ready-to-use sample face, speech, and text assets
│   └── create_sample_assets.py        # Demo asset generation script
├── src/
│   ├── __init__.py
│   ├── config.py                      # Package configuration bridge
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── face_processor.py          # OpenCV face detection, crop, resize (224x224), norm
│   │   ├── audio_processor.py         # 16kHz resampling, STFT, MFCCs, Mel-spectrogram
│   │   └── text_processor.py          # Tokenizer, cleaning, padding, attention masks
│   ├── models/
│   │   ├── __init__.py
│   │   ├── facial_encoder.py          # Deep Residual CNN visual backbone (512-dim)
│   │   ├── speech_encoder.py          # 1D-CNN + BiLSTM + Attention acoustic backbone (256-dim)
│   │   ├── text_encoder.py            # Transformer Multi-Head Encoder backbone (768-dim)
│   │   ├── fusion_modules.py          # Concatenation, Cross-Modal Attention, Hybrid Gating
│   │   └── multimodal_network.py      # Unified model supporting E1 through E7 & missing modalities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Multimodal Dataset, DataLoader & synthetic generator
│   │   └── trainer.py                 # AdamW optimizer, ReduceLROnPlateau, checkpointer
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py               # End-to-end inference engine with modality weights
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                 # Accuracy, Macro-F1, Confusion matrix, Table formatter
│       └── visualizer.py              # Matplotlib confusion matrix & loss curve plotting
├── experiments/
│   ├── run_all_experiments.py         # Automates benchmarks E1 through E7
│   └── benchmark_results.json         # Persisted experimental results
├── web_app/
│   ├── app.py                         # Flask web application prototype
│   ├── templates/
│   │   └── index.html                 # Modern responsive user interface
│   └── static/
│       ├── css/style.css              # Dark mode styling & gauges
│       └── js/main.js                 # Webcam snapshot, mic recording, AJAX predictions
└── saved_models/                      # Checkpoints (.pt)
```

---

## 🚀 6. Installation & Execution Guide

### 6.1 Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Virtual environment recommended:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 6.2 Generating Demo Assets & Testing Inference
Generate sample face, audio, and transcript inputs:
```bash
python data/create_sample_assets.py
```

Run test inference:
```bash
python -c "from src.inference.predictor import MultimodalEmotionPredictor; p = MultimodalEmotionPredictor(); print(p.predict(text_input='I am super excited!'))"
```

### 6.3 Running Research Benchmarks (Table I: E1 to E7)
Execute all 7 experimental setups to compute Accuracy, Macro-F1, and validate hypotheses:
```bash
python experiments/run_all_experiments.py
```
This produces a formatted comparison table and exports `experiments/benchmark_results.json`.

### 6.4 Launching the Web Prototype Demo
Launch the interactive web prototype:
```bash
python web_app/app.py
```
Open your browser at: **`http://localhost:5000`**

### 6.5 Pre-seeded Research Login Credentials
The system includes built-in role-based authentication with pre-seeded test credentials:

| Role | Email | Password |
|---|---|---|
| **Lead Researcher** | `sai.bhadre@indiraicem.ac.in` | `sai123` |
| **Project Guide** | `vaishali.hatkar@indiraicem.ac.in` | `guide123` |
| **External Reviewer** | `evaluator@sppu.ac.in` | `demo123` |

> *Note: New accounts can also be created in real time via the **"Create Account"** tab on `/login`.*

