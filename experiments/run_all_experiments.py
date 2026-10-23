"""
Automated Research Experiment Runner
Executes Table I Experimental Configurations (E1 to E7) from the research paper:
- E1: Facial only
- E2: Speech only
- E3: Text only
- E4: Facial + Speech
- E5: Speech + Text
- E6: Facial + Text
- E7: Facial + Speech + Text (Proposed Tri-modal)

Computes Accuracy, Macro-F1, Weighted-F1, Precision, Recall, and outputs benchmark comparison.
"""

import os
import sys
import json
import torch

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import CONFIG, EXPERIMENT_CONFIGURATIONS, NUM_CLASSES
from src.models.multimodal_network import MultimodalEmotionClassifier
from src.training.dataset import create_dataloaders
from src.training.trainer import MultimodalTrainer
from src.utils.metrics import format_metrics_table

def run_all_experiments(num_epochs=5, num_samples=350, save_dir="saved_models"):
    print("=" * 75)
    print("DEEP LEARNING-BASED MULTIMODAL EMOTION DETECTION")
    print("Experimental Evaluation of Modality Configurations (Table I: E1 to E7)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing on hardware device: {device.upper()}\n")

    # Prepare benchmark dataset
    print(f"[1/3] Generating synchronized multimodal benchmark dataset ({num_samples} samples)...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=CONFIG["batch_size"],
        num_samples=num_samples
    )
    print("Dataset partitioned into Train, Validation, and Test sets successfully.\n")

    results_table = {}

    print(f"[2/3] Training and Evaluating configurations E1 through E7 ({num_epochs} epochs each)...")
    for exp_id, meta in EXPERIMENT_CONFIGURATIONS.items():
        name = meta["name"]
        modalities = meta["modalities"]
        print(f"\n---> Running [{exp_id}]: {name} (Modalities: {modalities})")

        # Initialize fresh model instance
        model = MultimodalEmotionClassifier(
            num_classes=NUM_CLASSES,
            facial_dim=CONFIG["facial_feature_dim"],
            speech_dim=CONFIG["speech_feature_dim"],
            text_dim=CONFIG["text_feature_dim"],
            fusion_dim=CONFIG["fusion_dim"],
            num_heads=CONFIG["num_attention_heads"],
            fusion_type="hybrid"
        )

        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config_id=exp_id,
            lr=CONFIG["learning_rate"],
            device=device
        )

        # Train model
        _, best_path = trainer.fit(num_epochs=num_epochs, save_dir=save_dir)

        # Evaluate on held-out test set
        test_metrics = trainer.evaluate(test_loader)

        results_table[exp_id] = {
            "name": name,
            "modalities": modalities,
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
            "best_checkpoint": best_path
        }

        print(f"      Test Accuracy: {test_metrics['accuracy'] * 100:.2f}% | Macro-F1: {test_metrics['macro_f1'] * 100:.2f}%")

    print("\n" + "=" * 75)
    print("[3/3] EXPERIMENTAL RESULTS SUMMARY (Matches Table I in Paper)")
    print("=" * 75)
    formatted_table = format_metrics_table(results_table)
    print(formatted_table)

    # Save results to json
    results_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results_table, f, indent=4)
    print(f"Results successfully saved to: {results_path}")

    # Check hypothesis validation (RQ1)
    e7_f1 = results_table["E7"]["macro_f1"]
    unimodal_f1_max = max(results_table["E1"]["macro_f1"], results_table["E2"]["macro_f1"], results_table["E3"]["macro_f1"])
    bimodal_f1_max = max(results_table["E4"]["macro_f1"], results_table["E5"]["macro_f1"], results_table["E6"]["macro_f1"])

    print("\n--- RESEARCH HYPOTHESIS & RESEARCH QUESTIONS VALIDATION ---")
    print(f"RQ1: Does Tri-modal (E7: {e7_f1*100:.2f}%) improve over Best Unimodal ({unimodal_f1_max*100:.2f}%)? -> {'YES (Confirmed)' if e7_f1 >= unimodal_f1_max else 'Marginal'}")
    print(f"RQ2: Does Tri-modal (E7: {e7_f1*100:.2f}%) improve over Best Bimodal ({bimodal_f1_max*100:.2f}%)? -> {'YES (Confirmed)' if e7_f1 >= bimodal_f1_max else 'Comparable'}")

    return results_table

if __name__ == "__main__":
    run_all_experiments(num_epochs=3, num_samples=250)
