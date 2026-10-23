"""
Multimodal Emotion Inference Predictor
Combines preprocessors and trained network to perform real-time or offline inference
from face images, audio waveforms, and text strings.
"""

import os
import torch
import numpy as np
from config.config import CONFIG, EMOTION_CLASSES, NUM_CLASSES
from ..preprocessing.face_processor import FacialPreprocessor
from ..preprocessing.audio_processor import AudioPreprocessor
from ..preprocessing.text_processor import TextPreprocessor
from ..models.multimodal_network import MultimodalEmotionClassifier

class MultimodalEmotionPredictor:
    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        self.face_processor = FacialPreprocessor(target_size=CONFIG["image_size"])
        self.audio_processor = AudioPreprocessor(target_sr=CONFIG["audio_sample_rate"], duration_sec=CONFIG["audio_duration_sec"])
        self.text_processor = TextPreprocessor(max_length=CONFIG["text_max_length"])

        # Initialize network
        self.model = MultimodalEmotionClassifier(
            num_classes=NUM_CLASSES,
            facial_dim=CONFIG["facial_feature_dim"],
            speech_dim=CONFIG["speech_feature_dim"],
            text_dim=CONFIG["text_feature_dim"],
            fusion_dim=CONFIG["fusion_dim"],
            num_heads=CONFIG["num_attention_heads"],
            fusion_type="hybrid"
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, face_input=None, audio_input=None, text_input=None, config_id="E7"):
        """
        Runs multimodal emotion inference.
        Returns detailed prediction dictionary including fused emotion, confidence,
        modality contribution weights, and class probabilities.
        """
        face_tensor = None
        speech_tensor = None
        text_ids_tensor = None
        text_mask_tensor = None

        active_mods = []

        # 1. Process Face
        if face_input is not None:
            try:
                face_chw = self.face_processor.transform(face_input)
                face_tensor = torch.from_numpy(face_chw).unsqueeze(0).to(self.device) # [1, 3, 224, 224]
                active_mods.append("face")
            except Exception as e:
                print(f"[Warning] Facial processing error: {e}")

        # 2. Process Audio
        if audio_input is not None:
            try:
                speech_feat = self.audio_processor.extract_features(audio_input)
                speech_tensor = torch.from_numpy(speech_feat).unsqueeze(0).to(self.device) # [1, T, 40]
                active_mods.append("speech")
            except Exception as e:
                print(f"[Warning] Audio processing error: {e}")

        # 3. Process Text
        if text_input is not None and str(text_input).strip():
            try:
                t_ids, t_mask = self.text_processor.encode(text_input)
                text_ids_tensor = torch.from_numpy(t_ids).unsqueeze(0).to(self.device)
                text_mask_tensor = torch.from_numpy(t_mask).unsqueeze(0).to(self.device)
                active_mods.append("text")
            except Exception as e:
                print(f"[Warning] Text processing error: {e}")

        # Auto-detect config if not explicitly forced
        if config_id == "E7" and len(active_mods) < 3:
            if "face" in active_mods and "speech" in active_mods and "text" in active_mods:
                config_id = "E7"
            elif "face" in active_mods and "speech" in active_mods:
                config_id = "E4"
            elif "speech" in active_mods and "text" in active_mods:
                config_id = "E5"
            elif "face" in active_mods and "text" in active_mods:
                config_id = "E6"
            elif "face" in active_mods:
                config_id = "E1"
            elif "speech" in active_mods:
                config_id = "E2"
            elif "text" in active_mods:
                config_id = "E3"
            else:
                # Default fallback
                config_id = "E7"

        # Model Inference
        with torch.no_grad():
            logits, weights = self.model(
                face_input=face_tensor,
                speech_input=speech_tensor,
                text_ids=text_ids_tensor,
                text_mask=text_mask_tensor,
                config_id=config_id
            )

            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            predicted_emotion = EMOTION_CLASSES[pred_idx]
            confidence = float(probs[pred_idx])

            # Modality contribution weights (Section VII Dynamic Gating)
            mod_weights = weights.squeeze(0).cpu().numpy()
            face_wt = float(mod_weights[0])
            speech_wt = float(mod_weights[1])
            text_wt = float(mod_weights[2])

        # All emotion scores
        all_emotions = {EMOTION_CLASSES[i]: float(probs[i]) for i in range(len(EMOTION_CLASSES))}

        return {
            "predicted_emotion": predicted_emotion,
            "confidence": round(confidence * 100, 2),
            "config_id": config_id,
            "active_modalities": active_mods,
            "modality_contributions": {
                "facial": round(face_wt * 100, 2),
                "speech": round(speech_wt * 100, 2),
                "text": round(text_wt * 100, 2)
            },
            "class_probabilities": all_emotions
        }
