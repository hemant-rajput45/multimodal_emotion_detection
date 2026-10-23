"""
Speech Modality Encoder (E_s)
Acoustic Representation Network combining 1D Temporal Convolutions,
Bidirectional LSTM, and Attention Pooling as described in Section VI-B of the research paper.
Extracts representation vector F_s = E_s(X_s) of dimension 256.
"""

import torch
import torch.nn as nn

class AcousticAttentionPooling(nn.Module):
    """Learned attention pooling across temporal acoustic frames"""
    def __init__(self, hidden_dim):
        super(AcousticAttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, time_steps, hidden_dim]
        weights = self.attn(x) # [batch_size, time_steps, 1]
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1) # [batch_size, hidden_dim]
        return pooled, weights

class SpeechEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=256, num_layers=2):
        super(SpeechEncoder, self).__init__()
        self.output_dim = output_dim

        # 1D Temporal Convolutions to extract local spectral dynamics
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Bidirectional Recurrent Layer (BiLSTM) for long-range prosodic context
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Attention pooling over temporal sequence
        self.attn_pool = AcousticAttentionPooling(hidden_dim * 2)

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        """
        Input: x tensor of shape [batch_size, time_steps, input_dim]
        Output: F_s tensor of shape [batch_size, 256]
        """
        # Conv1d expects [batch_size, channels, time_steps]
        x = x.transpose(1, 2)
        feat = self.conv_layers(x)
        feat = feat.transpose(1, 2) # back to [batch_size, reduced_time, channels]

        lstm_out, _ = self.bilstm(feat)
        pooled, _ = self.attn_pool(lstm_out)
        F_s = self.fc(pooled)
        return F_s
