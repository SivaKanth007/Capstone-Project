"""
LSTM Failure Probability Predictor
====================================
Binary classifier: predicts probability of failure within h cycles.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class LSTMPredictor(nn.Module):
    """
    LSTM-based binary classifier for failure prediction.

    Architecture:
    - 2-layer LSTM with attention mechanism
    - Dense layers with dropout → Sigmoid output

    Output: P(failure within h cycles)
    """

    def __init__(self, input_dim, hidden_dim=None, num_layers=None, dropout=None):
        super().__init__()

        self.hidden_dim = hidden_dim or config.PRED_HIDDEN_DIM
        self.num_layers = num_layers or config.PRED_NUM_LAYERS
        self.dropout = dropout or config.PRED_DROPOUT

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.input_dim = input_dim

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden_dim)

        # Classification
        output = self.classifier(context)  # (batch, 1)
        return output.squeeze(-1), attn_weights.squeeze(-1)

    def predict_proba(self, x):
        """Get failure probability for input sequences."""
        self.eval()
        with torch.no_grad():
            x = x.to(config.DEVICE)
            proba, attn = self.forward(x)
            return proba.cpu().numpy(), attn.cpu().numpy()


class PredictorTrainer:
    """Training loop for LSTM failure predictor with class balancing."""

    def __init__(self, model, lr=None, epochs=None, batch_size=None):
        self.model = model.to(config.DEVICE)
        self.lr = lr or config.PRED_LEARNING_RATE
        self.epochs = epochs or config.PRED_EPOCHS
        self.batch_size = batch_size or config.PRED_BATCH_SIZE
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.train_history = []
        self.val_history = []

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the failure predictor.

        Parameters
        ----------
        X_train, y_train : np.ndarray — training sequences and binary labels
        X_val, y_val : np.ndarray, optional — validation data
        """
        # Compute class weights for imbalanced data
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(config.DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Actually we use BCE since we already have sigmoid, so use weight differently
        criterion = nn.BCELoss(reduction='none')

        train_tensor_x = torch.FloatTensor(X_train)
        train_tensor_y = torch.FloatTensor(y_train)
        train_loader = DataLoader(
            TensorDataset(train_tensor_x, train_tensor_y),
            batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                batch_size=self.batch_size, shuffle=False
            )

        best_val_f1 = 0
        best_state = None
        weight_pos = neg_count / max(pos_count, 1)

        print(f"\n[PREDICTOR] Training on {config.DEVICE} "
              f"({len(X_train)} samples, pos_rate={pos_count/len(y_train):.2%})")
        print(f"[PREDICTOR] Positive weight: {weight_pos:.2f}")

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)

                proba, _ = self.model(batch_x)
                loss_per_sample = criterion(proba, batch_y)

                # Apply class weights
                weights = torch.where(batch_y == 1, weight_pos, 1.0)
                loss = (loss_per_sample * weights).mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item() * len(batch_x)

            train_loss /= len(X_train)
            self.train_history.append(train_loss)

            # Validation
            if val_loader is not None and (epoch + 1) % 5 == 0:
                metrics = self._evaluate(val_loader, y_val)
                self.val_history.append(metrics)
                self.scheduler.step(1 - metrics["f1"])

                if metrics["f1"] > best_val_f1:
                    best_val_f1 = metrics["f1"]
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                print(f"  Epoch {epoch+1}/{self.epochs} — Loss: {train_loss:.4f} | "
                      f"F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f} | "
                      f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f}")
            elif (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} — Loss: {train_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(config.DEVICE)
            print(f"[PREDICTOR] Restored best model (F1={best_val_f1:.4f})")

        return self.model

    def _evaluate(self, loader, y_true):
        """Evaluate model on validation/test set."""
        self.model.eval()
        all_proba = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(config.DEVICE)
                proba, _ = self.model(batch_x)
                all_proba.extend(proba.cpu().numpy())

        y_proba = np.array(all_proba)
        y_pred = (y_proba >= 0.5).astype(int)

        return {
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        }

    def save_model(self, filepath=None):
        """Save trained model."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "lstm_predictor.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }, filepath)
        print(f"[PREDICTOR] Model saved to {filepath}")


def load_predictor(filepath=None):
    """Load a trained predictor."""
    filepath = filepath or os.path.join(config.MODELS_DIR, "lstm_predictor.pt")
    checkpoint = torch.load(filepath, map_location=config.DEVICE, weights_only=False)

    model = LSTMPredictor(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(config.DEVICE)
    model.eval()

    print(f"[PREDICTOR] Loaded model from {filepath}")
    return model
