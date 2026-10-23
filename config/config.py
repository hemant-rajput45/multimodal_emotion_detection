"""
Centralized Configuration for Multimodal Emotion Detection
Title: Deep Learning-Based Multimodal Emotion Detection: Fusion of Facial, Speech, and Text Modalities
Student: Sai Bhadre (MCA)
"""

import os

# Emotion Categories (7 standard basic emotion categories as stated in research proposal)
EMOTION_CLASSES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Sadness",
    "Surprise",
    "Neutral"
]

NUM_CLASSES = len(EMOTION_CLASSES)
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_CLASSES)}
IDX_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTION_CLASSES)}

# Feature and Model Dimensions
CONFIG = {
    # Feature Dimensions from Modality-Specific Encoders
    "facial_feature_dim": 512,      # Visual representation (ResNet/EfficientNet representation)
    "speech_feature_dim": 256,      # Acoustic representation (HuBERT / MFCC+Spectrogram BiLSTM)
    "text_feature_dim": 768,        # Language contextual representation (RoBERTa / BERT)

    # Common Projected Dimension for Multimodal Fusion Space (Equations 6, 7, 8)
    "fusion_dim": 256,

    # Cross-Modal Attention Parameters (Equation 10)
    "num_attention_heads": 4,
    "attention_dropout": 0.2,

    # Classification Head
    "classifier_hidden_dim": 128,
    "classifier_dropout": 0.3,
    "num_classes": NUM_CLASSES,

    # Training Parameters
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_epochs": 20,
    "device": "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",

    # Preprocessing Parameters
    "image_size": (224, 224),
    "audio_sample_rate": 16000,
    "audio_duration_sec": 3.0,
    "text_max_length": 64
}

# Experimental Configurations (Table I in Research Paper)
EXPERIMENT_CONFIGURATIONS = {
    "E1": {"name": "Facial only", "modalities": ["face"]},
    "E2": {"name": "Speech only", "modalities": ["speech"]},
    "E3": {"name": "Text only", "modalities": ["text"]},
    "E4": {"name": "Facial + Speech", "modalities": ["face", "speech"]},
    "E5": {"name": "Speech + Text", "modalities": ["speech", "text"]},
    "E6": {"name": "Facial + Text", "modalities": ["face", "text"]},
    "E7": {"name": "Facial + Speech + Text", "modalities": ["face", "speech", "text"]}
}
