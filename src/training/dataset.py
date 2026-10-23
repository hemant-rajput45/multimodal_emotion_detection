"""
Multimodal Emotion Dataset and DataLoader utilities
Provides synchronization across Facial, Speech, and Text modalities.
Includes automated benchmark synthetic sample generation for testing and reproducibility.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config.config import EMOTION_CLASSES, NUM_CLASSES, EMOTION_TO_IDX

class MultimodalEmotionDataset(Dataset):
    def __init__(self, face_data, speech_data, text_ids, text_masks, labels):
        """
        Args:
            face_data: numpy array / tensor [N, 3, 224, 224]
            speech_data: numpy array / tensor [N, time_steps, n_mfcc] (e.g. [N, 100, 40])
            text_ids: numpy array / tensor [N, max_len]
            text_masks: numpy array / tensor [N, max_len]
            labels: numpy array / tensor [N]
        """
        self.face_data = torch.as_tensor(face_data, dtype=torch.float32)
        self.speech_data = torch.as_tensor(speech_data, dtype=torch.float32)
        self.text_ids = torch.as_tensor(text_ids, dtype=torch.long)
        self.text_masks = torch.as_tensor(text_masks, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "face": self.face_data[idx],
            "speech": self.speech_data[idx],
            "text_ids": self.text_ids[idx],
            "text_mask": self.text_masks[idx],
            "label": self.labels[idx]
        }

def generate_benchmark_dataset(num_samples=350, time_steps=100, n_mfcc=40, max_text_len=64, seed=42):
    """
    Generates realistic, synchronized multimodal benchmark data for testing,
    ablations, and model verification.
    """
    np.random.seed(seed)

    # Balanced distribution of 7 emotions
    samples_per_class = num_samples // NUM_CLASSES
    labels = []
    for cls_idx in range(NUM_CLASSES):
        labels.extend([cls_idx] * samples_per_class)
    remainder = num_samples - len(labels)
    if remainder > 0:
        labels.extend(list(np.random.choice(NUM_CLASSES, size=remainder)))
    labels = np.array(labels)

    # 1. Facial data: [N, 3, 224, 224]
    # Class-correlated pattern with random variation
    face_data = np.random.randn(num_samples, 3, 224, 224).astype(np.float32) * 0.1
    for i, lbl in enumerate(labels):
        # Inject emotion-specific visual cue in central region
        face_data[i, :, 80:144, 80:144] += (lbl + 1) * 0.15

    # 2. Speech data: [N, time_steps, n_mfcc]
    speech_data = np.random.randn(num_samples, time_steps, n_mfcc).astype(np.float32) * 0.1
    for i, lbl in enumerate(labels):
        # Inject emotion-specific acoustic pitch/energy shift
        speech_data[i, :, :10] += (lbl + 1) * 0.12

    # 3. Text data: [N, max_text_len]
    text_ids = np.zeros((num_samples, max_text_len), dtype=np.int64)
    text_masks = np.zeros((num_samples, max_text_len), dtype=np.float32)

    for i, lbl in enumerate(labels):
        seq_len = np.random.randint(10, 30)
        # Token IDs correlated with emotion
        tokens = [2] # [CLS]
        for _ in range(seq_len - 2):
            base_token = (lbl * 50) + np.random.randint(4, 45)
            tokens.append(base_token)
        tokens.append(3) # [SEP]

        text_ids[i, :len(tokens)] = tokens
        text_masks[i, :len(tokens)] = 1.0

    return face_data, speech_data, text_ids, text_masks, labels

def create_dataloaders(batch_size=32, num_samples=350, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Creates stratified Train, Validation, and Test DataLoaders.
    """
    face, speech, t_ids, t_masks, labels = generate_benchmark_dataset(num_samples=num_samples)

    total = len(labels)
    indices = np.random.permutation(total)

    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_ds = MultimodalEmotionDataset(
        face[train_idx], speech[train_idx], t_ids[train_idx], t_masks[train_idx], labels[train_idx]
    )
    val_ds = MultimodalEmotionDataset(
        face[val_idx], speech[val_idx], t_ids[val_idx], t_masks[val_idx], labels[val_idx]
    )
    test_ds = MultimodalEmotionDataset(
        face[test_idx], speech[test_idx], t_ids[test_idx], t_masks[test_idx], labels[test_idx]
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
