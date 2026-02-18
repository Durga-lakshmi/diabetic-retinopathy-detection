import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, balanced_accuracy_score, cohen_kappa_score
)

class BinaryMetrics:
    def __init__(self, threshold: float = 0.4, zero_division: int = 0, do_threshold_sweep: bool = False):
        self.t = threshold
        self.zero_division = zero_division
        self.do_threshold_sweep = do_threshold_sweep

    def __call__(self, y_true_np: np.ndarray, y_score_np: np.ndarray) -> dict:
        # y_score_np expected shape [N,1] or [N,]
        y_score_np = y_score_np.reshape(-1)
        y_true_np = y_true_np.reshape(-1)

        auc = roc_auc_score(y_true_np, y_score_np)
        pred = (y_score_np >= self.t).astype(np.int64)

        return {
            "auc": float(auc),
            "accuracy": float(accuracy_score(y_true_np, pred)),
            "f1": float(f1_score(y_true_np, pred, zero_division=self.zero_division)),
            "precision": float(precision_score(y_true_np, pred, zero_division=self.zero_division)),
            "recall": float(recall_score(y_true_np, pred, zero_division=self.zero_division)),
            "confusion_matrix": confusion_matrix(y_true_np, pred),
            "threshold": float(self.t),
        }


class FiveClassMetrics:
    def __init__(self, include_confusion_matrix: bool = True):
        self.include_confusion_matrix = include_confusion_matrix

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        # logits [N,5], targets [N]
        if isinstance(logits, torch.Tensor):
            print("[DEBUG] logits type:", type(logits))
            if hasattr(logits, "shape"):
                print("[DEBUG] logits shape:", logits.shape)
            else:
                print("[DEBUG] logits:", logits)
            y_pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        else:
            y_pred = np.argmax(logits, axis=1)

        if isinstance(targets, torch.Tensor):
            y_true = targets.cpu().numpy()
        else:
            y_true = targets

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        out = {
            "acc": float(acc),
            "balanced_acc": float(bal_acc),
            "qwk": float(qwk),
            "macro_f1": float(macro_f1),
        }
        if self.include_confusion_matrix:
            out["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        return out
