"""
Text Modality Encoder (E_t)
Transformer-Based Language Encoder architecture (RoBERTa / BERT style)
as described in Section VI-C of the research paper.
Extracts representation vector F_t = E_t(X_t) of dimension 768.
"""

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=256, num_heads=4, hidden_dim=512, num_layers=3, output_dim=768, max_len=64):
        super(TextEncoder, self).__init__()
        self.output_dim = output_dim

        # Token Embedding and Positional Encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        self.embed_dropout = nn.Dropout(p=0.1)

        # Multi-layer Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Contextual Pooling and Projection to 768-dim space
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Input:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1 for valid token, 0 for padding)
        Output:
            F_t: [batch_size, 768]
        """
        # Key padding mask: True for positions to be masked (padding)
        src_key_padding_mask = (input_ids == 0) if attention_mask is None else (attention_mask == 0)

        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.embed_dropout(x)

        transformed = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Extract representation: [CLS] token representation at index 0, or masked mean
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            sum_embeddings = torch.sum(transformed * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = transformed[:, 0, :]

        F_t = self.fc(pooled)
        return F_t
