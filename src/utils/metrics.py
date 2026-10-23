"""
Evaluation Metrics & Reporting
Implements:
- Accuracy (Equation 12)
- Precision, Recall, Macro-F1, Weighted-F1
- Confusion Matrix
- Experimental Result Table generator
as described in Section VIII-A of the research paper.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred, emotion_classes=None):
    """
    Computes all classification metrics required by the research paper.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    results = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "macro_f1": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "confusion_matrix": cm.tolist()
    }
    return results

def format_metrics_table(experiment_results):
    """
    Formats multi-experiment comparison (Table I: E1 to E7) into clean markdown table.
    """
    header = "| Exp ID | Configuration | Modalities | Accuracy | Macro-F1 | Weighted-F1 | Precision | Recall |\n"
    divider = "|--------|---------------|------------|----------|----------|-------------|-----------|--------|\n"
    rows = []

    for exp_id, data in sorted(experiment_results.items()):
        name = data.get("name", "N/A")
        mods = "+".join(data.get("modalities", []))
        acc = f"{data.get('accuracy', 0.0) * 100:.2f}%"
        macro_f1 = f"{data.get('macro_f1', 0.0) * 100:.2f}%"
        weighted_f1 = f"{data.get('weighted_f1', 0.0) * 100:.2f}%"
        prec = f"{data.get('precision_macro', 0.0) * 100:.2f}%"
        rec = f"{data.get('recall_macro', 0.0) * 100:.2f}%"
        rows.append(f"| {exp_id} | {name} | {mods} | {acc} | {macro_f1} | {weighted_f1} | {prec} | {rec} |")

    return header + divider + "\n".join(rows) + "\n"
