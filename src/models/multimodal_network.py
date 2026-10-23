"""
Unified Multimodal Emotion Recognition Network
Supports all 7 experimental configurations (E1 to E7) from Table I of the research paper:
- E1: Facial only
- E2: Speech only
- E3: Text only
- E4: Facial + Speech
- E5: Speech + Text
- E6: Facial + Text
- E7: Facial + Speech + Text (Proposed Tri-modal Architecture)
"""

import torch
import torch.nn as nn
from .facial_encoder import FacialEncoder
from .speech_encoder import SpeechEncoder
from .text_encoder import TextEncoder
from .fusion_modules import ModalityProjection, MultimodalFusion

class MultimodalEmotionClassifier(nn.Module):
    def __init__(
        self,
        num_classes=7,
        facial_dim=512,
        speech_dim=256,
        text_dim=768,
        fusion_dim=256,
        num_heads=4,
        fusion_type="hybrid",
        dropout=0.3
    ):
        super(MultimodalEmotionClassifier, self).__init__()
        self.num_classes = num_classes
        self.fusion_dim = fusion_dim
        self.fusion_type = fusion_type

        # 1. Modality-Specific Encoders (Section VI-A, VI-B, VI-C)
        self.face_encoder = FacialEncoder(output_dim=facial_dim)
        self.speech_encoder = SpeechEncoder(output_dim=speech_dim)
        self.text_encoder = TextEncoder(output_dim=text_dim)

        # 2. Projection Layers to Common Embedding Space (Section VI-D)
        self.proj_face = ModalityProjection(in_dim=facial_dim, out_dim=fusion_dim)
        self.proj_speech = ModalityProjection(in_dim=speech_dim, out_dim=fusion_dim)
        self.proj_text = ModalityProjection(in_dim=text_dim, out_dim=fusion_dim)

        # Learned Null/Mask Embeddings for Missing Modalities (RQ3 Robustness)
        self.null_face = nn.Parameter(torch.randn(1, fusion_dim))
        self.null_speech = nn.Parameter(torch.randn(1, fusion_dim))
        self.null_text = nn.Parameter(torch.randn(1, fusion_dim))

        # 3. Multimodal Fusion Module (Section VII)
        self.fusion_module = MultimodalFusion(dim=fusion_dim, num_heads=num_heads, fusion_type=fusion_type)

        # 4. Modality-specific Unimodal Classifiers (for Unimodal baselines E1, E2, E3)
        self.clf_face = nn.Linear(fusion_dim, num_classes)
        self.clf_speech = nn.Linear(fusion_dim, num_classes)
        self.clf_text = nn.Linear(fusion_dim, num_classes)

        # 5. Final Multimodal Classifier Head (Equation 5)
        in_clf_dim = fusion_dim * 3 if fusion_type == "concat" else fusion_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_clf_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def extract_modality_features(self, face_input=None, speech_input=None, text_ids=None, text_mask=None):
        """
        Extracts and projects features for each modality.
        Gracefully falls back to learned null representation if modality is missing/absent.
        """
        batch_size = None
        for item in [face_input, speech_input, text_ids]:
            if item is not None:
                batch_size = item.size(0)
                break
        if batch_size is None:
            batch_size = 1

        # Facial representation
        if face_input is not None:
            f_face = self.face_encoder(face_input)
            z_face = self.proj_face(f_face)
        else:
            z_face = self.null_face.repeat(batch_size, 1)

        # Speech representation
        if speech_input is not None:
            f_speech = self.speech_encoder(speech_input)
            z_speech = self.proj_speech(f_speech)
        else:
            z_speech = self.null_speech.repeat(batch_size, 1)

        # Text representation
        if text_ids is not None:
            f_text = self.text_encoder(text_ids, text_mask)
            z_text = self.proj_text(f_text)
        else:
            z_text = self.null_text.repeat(batch_size, 1)

        return z_face, z_speech, z_text

    def forward(self, face_input=None, speech_input=None, text_ids=None, text_mask=None, config_id="E7"):
        """
        Executes forward pass according to configuration ID (E1 to E7):
        - E1: Facial only
        - E2: Speech only
        - E3: Text only
        - E4: Facial + Speech
        - E5: Speech + Text
        - E6: Facial + Text
        - E7: Tri-modal (Facial + Speech + Text)
        Returns:
            logits: [batch_size, num_classes]
            modality_weights: [batch_size, 3] (contribution of face, speech, text)
        """
        z_face, z_speech, z_text = self.extract_modality_features(
            face_input=face_input if "face" in self._get_active_modalities(config_id) else None,
            speech_input=speech_input if "speech" in self._get_active_modalities(config_id) else None,
            text_ids=text_ids if "text" in self._get_active_modalities(config_id) else None,
            text_mask=text_mask if "text" in self._get_active_modalities(config_id) else None
        )

        batch_size = z_face.size(0)

        # Unimodal direct classification
        if config_id == "E1":
            logits = self.clf_face(z_face)
            weights = torch.tensor([[1.0, 0.0, 0.0]], device=z_face.device).repeat(batch_size, 1)
            return logits, weights
        elif config_id == "E2":
            logits = self.clf_speech(z_speech)
            weights = torch.tensor([[0.0, 1.0, 0.0]], device=z_speech.device).repeat(batch_size, 1)
            return logits, weights
        elif config_id == "E3":
            logits = self.clf_text(z_text)
            weights = torch.tensor([[0.0, 0.0, 1.0]], device=z_text.device).repeat(batch_size, 1)
            return logits, weights

        # Multimodal Fusion (Bimodal & Tri-modal)
        fused_rep, weights = self.fusion_module(z_face, z_speech, z_text)
        logits = self.classifier(fused_rep)
        return logits, weights

    def _get_active_modalities(self, config_id):
        mapping = {
            "E1": ["face"],
            "E2": ["speech"],
            "E3": ["text"],
            "E4": ["face", "speech"],
            "E5": ["speech", "text"],
            "E6": ["face", "text"],
            "E7": ["face", "speech", "text"]
        }
        return mapping.get(config_id, ["face", "speech", "text"])
