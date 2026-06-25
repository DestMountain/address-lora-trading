"""
Evaluation metrics for address-level behavior models.
"""

import numpy as np
from typing import Any


def compute_metrics(
    y_true_action: np.ndarray,
    y_pred_action: np.ndarray,
    y_true_price: np.ndarray | None = None,
    y_pred_price: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true_action: (n,) true action labels.
        y_pred_action: (n,) predicted action labels.
        y_true_price: (n,) true price deltas (optional).
        y_pred_price: (n,) predicted price deltas (optional).

    Returns:
        dict of metric name -> value.
    """
    metrics: dict[str, float] = {}

    # Direction accuracy
    correct = (y_pred_action == y_true_action).sum()
    total = len(y_true_action)
    metrics["direction_accuracy"] = float(correct / max(total, 1))
    metrics["total_samples"] = total

    # Per-action accuracy
    from ..data.preprocess import NUM_ACTIONS
    for a in range(NUM_ACTIONS):
        mask = y_true_action == a
        if mask.sum() > 0:
            acc = (y_pred_action[mask] == y_true_action[mask]).mean()
            metrics[f"acc_action_{a}"] = float(acc)

    # Price prediction metrics
    if y_true_price is not None and y_pred_price is not None:
        abs_errors = np.abs(y_pred_price - y_true_price)
        metrics["price_mae"] = float(abs_errors.mean())
        metrics["price_rmse"] = float(np.sqrt(((y_pred_price - y_true_price) ** 2).mean()))

    # Confusion matrix summary
    # Per-class precision, recall, f1
    from sklearn.metrics import precision_recall_fscore_support
    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_action, y_pred_action, average=None, zero_division=0
        )
        for a in range(len(precision)):
            if support[a] > 0:
                metrics[f"precision_{a}"] = float(precision[a])
                metrics[f"recall_{a}"] = float(recall[a])
                metrics[f"f1_{a}"] = float(f1[a])

        # Macro averages
        metrics["macro_precision"] = float(np.mean(precision))
        metrics["macro_recall"] = float(np.mean(recall))
        metrics["macro_f1"] = float(np.mean(f1))
    except Exception:
        pass

    return metrics


def print_metrics_table(metrics: dict[str, Any], title: str = "Results"):
    """Print metrics as a formatted table."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")

    key_metrics = [
        "direction_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "price_mae",
        "price_rmse",
        "total_samples",
    ]
    for k in key_metrics:
        if k in metrics:
            print(f"  {k:20s}: {metrics[k]:.4f}")

    print(f"{'=' * 50}\n")
