"""
Model Training Module
Implements training loop, validation, loss calculation, and checkpoint saving
for any experimental configuration (E1 through E7).
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from ..utils.metrics import compute_metrics

class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader, config_id="E7", lr=1e-4, weight_decay=1e-5, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config_id = config_id
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": []
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            face = batch["face"].to(self.device)
            speech = batch["speech"].to(self.device)
            text_ids = batch["text_ids"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            logits, _ = self.model(
                face_input=face,
                speech_input=speech,
                text_ids=text_ids,
                text_mask=text_mask,
                config_id=self.config_id
            )

            loss = self.criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * face.size(0)

        epoch_loss = total_loss / len(self.train_loader.dataset)
        return epoch_loss

    def evaluate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                face = batch["face"].to(self.device)
                speech = batch["speech"].to(self.device)
                text_ids = batch["text_ids"].to(self.device)
                text_mask = batch["text_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits, _ = self.model(
                    face_input=face,
                    speech_input=speech,
                    text_ids=text_ids,
                    text_mask=text_mask,
                    config_id=self.config_id
                )

                loss = self.criterion(logits, labels)
                total_loss += loss.item() * face.size(0)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = avg_loss
        return metrics

    def fit(self, num_epochs=10, save_dir=None):
        best_val_f1 = -1.0
        best_model_path = None

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]

            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_f1)

            # Save best checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                if save_dir:
                    best_model_path = os.path.join(save_dir, f"best_model_{self.config_id}.pt")
                    torch.save(self.model.state_dict(), best_model_path)

        return self.history, best_model_path
