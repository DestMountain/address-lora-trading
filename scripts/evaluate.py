#!/usr/bin/env python3
"""
Evaluate trained models and print metrics.

Usage:
    python scripts/evaluate.py [--model checkpoints/lstm_best.pt] [--data data/raw]
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import fills_to_sequences, create_sequences_for_training
from src.models.lstm_model import TradeLSTM, evaluate
from src.models.baseline import MarkovModel
from src.eval.metrics import compute_metrics, print_metrics_table


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", type=str, default="checkpoints/lstm_best.pt")
    parser.add_argument("--data", type=str, default="data/raw")
    parser.add_argument("--seq-length", type=int, default=32)
    args = parser.parse_args()

    # Load data via train.py's loader
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import load_data

    sequences = load_data(args.data)
    if not sequences:
        return

    print(f"\n[*] Loaded {len(sequences)} sequences for evaluation")

    # Create test sequences
    X, y_action, y_price, y_size = create_sequences_for_training(
        sequences, seq_length=args.seq_length
    )
    print(f"    {len(X)} test samples")

    # Evaluate
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y_action),
        torch.FloatTensor(y_price),
        torch.FloatTensor(y_size),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Load LSTM model
    model_path = Path(args.model)
    if model_path.exists():
        print(f"\n[*] Loading LSTM from {model_path}")
        model = TradeLSTM(input_dim=X.shape[2])
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        metrics = evaluate(model, loader, torch.device("cpu"))
        print_metrics_table(metrics, "LSTM Model")

        # Also compute per-action metrics
        all_preds = []
        all_true = []
        all_price_pred = []
        all_price_true = []

        with torch.no_grad():
            for bx, by_a, by_p, by_sz in loader:
                logits, pp, ps = model(bx)
                all_preds.extend(logits.argmax(dim=-1).numpy())
                all_true.extend(by_a.numpy())
                all_price_pred.extend(pp.squeeze().numpy())
                all_price_true.extend(by_p.numpy())

        detailed = compute_metrics(
            np.array(all_true),
            np.array(all_preds),
            np.array(all_price_true),
            np.array(all_price_pred),
        )
        print_metrics_table(detailed, "Detailed Metrics")
    else:
        print(f"\n[!] Model not found at {model_path}")
        print("    Train first: `python scripts/train.py --model lstm`")


if __name__ == "__main__":
    main()
