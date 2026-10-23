"""
Visualization utility for research figures:
- Confusion Matrix Heatmaps
- Training Loss / Accuracy curves
- Modality Contribution radar / bar charts
"""

import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plots and saves confusion matrix heatmap as PNG for the research paper.
    """
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=int(cm[i, j]), va='center', ha='center', size='medium',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='left')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Emotion Label', fontweight='bold', labelpad=10)
    ax.set_ylabel('True Ground Truth Label', fontweight='bold', labelpad=10)
    ax.set_title('Confusion Matrix - Multimodal Emotion Recognition (E7)', fontweight='bold', pad=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def plot_training_curves(history, save_path):
    """
    Plots training and validation loss & accuracy curves over epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    ax1.set_title('Cross-Entropy Loss vs. Epochs', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy / F1 curve
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='#2ca02c', linewidth=2)
    ax2.plot(history['val_macro_f1'], label='Validation Macro-F1', color='#d62728', linewidth=2, linestyle='--')
    ax2.set_title('Performance Metrics vs. Epochs', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
