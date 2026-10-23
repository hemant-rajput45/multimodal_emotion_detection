"""
Multimodal Fusion Modules
Implements:
1. Feature Projection Layers (Equations 6, 7, 8)
2. Early Concatenation Fusion Baseline (Equation 9)
3. Cross-Modal Multi-Head Attention Fusion (Equation 10)
4. Hybrid / Gated Multimodal Fusion with Modality Contribution Weights (Equation 11)
as defined in Sections VI-D and VII of the research paper.
"""

import math
import torch
import torch.nn as nn

class ModalityProjection(nn.Module):
    """
    Projects modality-specific features (F_f, F_s, F_t) into common embedding space:
    Z_f = W_f F_f + b_f
    Z_s = W_s F_s + b_s
    Z_t = W_t F_t + b_t
    """
    def __init__(self, in_dim, out_dim=256, dropout=0.2):
        super(ModalityProjection, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.project(x)

class CrossModalAttention(nn.Module):
    """
    Multi-Head Cross-Modal Attention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    Allows modality A to attend to modality B.
    """
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query, key, value shape: [batch_size, 1, dim]
        attn_out, attn_weights = self.mha(query, key, value)
        out = self.norm(query + self.dropout(attn_out))
        return out, attn_weights

class MultimodalFusion(nn.Module):
    """
    Proposed Hybrid Fusion Architecture with Cross-Modal Attention and Adaptive Gating
    """
    def __init__(self, dim=256, num_heads=4, fusion_type="hybrid"):
        super(MultimodalFusion, self).__init__()
        self.dim = dim
        self.fusion_type = fusion_type  # "concat", "attention", "hybrid"

        # Pairwise Cross-Modal Attention blocks
        # Face attending to Speech & Text
        self.cross_face_speech = CrossModalAttention(dim, num_heads)
        self.cross_face_text = CrossModalAttention(dim, num_heads)

        # Speech attending to Face & Text
        self.cross_speech_face = CrossModalAttention(dim, num_heads)
        self.cross_speech_text = CrossModalAttention(dim, num_heads)

        # Text attending to Face & Speech
        self.cross_text_face = CrossModalAttention(dim, num_heads)
        self.cross_text_speech = CrossModalAttention(dim, num_heads)

        # Self-Attention across modalities
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)

        # Dynamic Gating Network to learn modality contribution weights
        self.gating = nn.Sequential(
            nn.Linear(dim * 3, 3),
            nn.Softmax(dim=-1)
        )

        # Fusion feedforward network
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )

    def forward(self, z_face, z_speech, z_text, active_modalities=None):
        """
        Inputs:
            z_face, z_speech, z_text: [batch_size, dim]
            active_modalities: list of active modality keys (e.g. ['face', 'speech', 'text'])
        Returns:
            fused_rep: [batch_size, dim]
            modality_weights: [batch_size, 3] (contribution of face, speech, text)
        """
        batch_size = z_face.size(0)

        # Reshape to sequence length 1: [batch_size, 1, dim]
        zf = z_face.unsqueeze(1)
        zs = z_speech.unsqueeze(1)
        zt = z_text.unsqueeze(1)

        if self.fusion_type == "concat":
            # Simple Concatenation Baseline (Equation 9)
            concat_rep = torch.cat([z_face, z_speech, z_text], dim=-1) # [batch, dim*3]
            weights = torch.ones(batch_size, 3, device=z_face.device) / 3.0
            return concat_rep, weights

        # 1. Pairwise Cross-Modal Context Interaction (Equation 10)
        # Face enhanced by Speech and Text
        f_from_s, _ = self.cross_face_speech(zf, zs, zs)
        f_from_t, _ = self.cross_face_text(zf, zt, zt)
        z_f_enhanced = (zf + f_from_s + f_from_t) / 3.0

        # Speech enhanced by Face and Text
        s_from_f, _ = self.cross_speech_face(zs, zf, zf)
        s_from_t, _ = self.cross_speech_text(zs, zt, zt)
        z_s_enhanced = (zs + s_from_f + s_from_t) / 3.0

        # Text enhanced by Face and Speech
        t_from_f, _ = self.cross_text_face(zt, zf, zf)
        t_from_s, _ = self.cross_text_speech(zt, zs, zs)
        z_t_enhanced = (zt + t_from_f + t_from_s) / 3.0

        # 2. Self-Attention layer across 3 modalities
        stacked = torch.cat([z_f_enhanced, z_s_enhanced, z_t_enhanced], dim=1) # [batch, 3, dim]
        sa_out, _ = self.self_attn(stacked, stacked, stacked)
        sa_out = self.self_norm(stacked + sa_out) # [batch, 3, dim]

        z_f_final = sa_out[:, 0, :]
        z_s_final = sa_out[:, 1, :]
        z_t_final = sa_out[:, 2, :]

        # 3. Dynamic Modality Gating / Hybrid Fusion
        concat_features = torch.cat([z_f_final, z_s_final, z_t_final], dim=-1) # [batch, dim*3]
        modality_weights = self.gating(concat_features) # [batch, 3]

        # Apply gating weights
        w_f = modality_weights[:, 0].unsqueeze(-1)
        w_s = modality_weights[:, 1].unsqueeze(-1)
        w_t = modality_weights[:, 2].unsqueeze(-1)

        weighted_combined = torch.cat([w_f * z_f_final, w_s * z_s_final, w_t * z_t_final], dim=-1)
        fused_rep = self.fusion_mlp(weighted_combined) # [batch, dim]

        return fused_rep, modality_weights
