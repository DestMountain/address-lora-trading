#!/usr/bin/env python3
"""
Train the next-trade prediction model.

Usage:
    python scripts/train.py [--model lstm] [--epochs 50] [--batch-size 64]
"""

import sys
import os
import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import (
    fills_to_sequences,
    create_sequences_for_training,
    TradeSequence,
)
from src.models.lstm_model import TradeLSTM, train_epoch, evaluate
from src.models.baseline import MarkovModel
from src.eval.metrics import compute_metrics, print_metrics_table


def load_data(data_dir: str) -> list[TradeSequence]:
    """Load raw data and convert to sequences."""
    data_dir = Path(data_dir)
    sequences = []

    if not data_dir.exists():
        print(f"[!] No data found at {data_dir}")
        print("    Run `python scripts/fetch_data.py` first!")
        return []

    json_files = list(data_dir.glob("0x*.json"))
    print(f"[*] Loading {len(json_files)} address files from {data_dir}...")

    for fpath in json_files:
        try:
            with open(fpath) as f:
                fills_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Skipping {fpath.name}: {e}")
            continue

        # Reconstruct Fill objects (dict -> Fill)
        # Reconstruct Fill objects (dict -> Fill)
        from src.data.hyperliquid import Fill
        fills = []
        for item in fills_data:
            if isinstance(item, dict):
                try:
                    fills.append(Fill(
                        coin=item.get("coin", ""),
                        px=float(item.get("px", 0)),
                        sz=float(item.get("sz", 0)),
                        side=item.get("side", ""),
                        time=int(item.get("time", 0)),
                        start_position=float(item.get("start_position", 0)),
                        dir=item.get("dir", ""),
                        closed_pnl=float(item.get("closed_pnl", 0)),
                        hash=item.get("hash", ""),
                        oid=int(item.get("oid", 0)),
                        fee=float(item.get("fee", 0)),
                        tid=int(item.get("tid", 0)),
                        fee_token=item.get("fee_token", ""),
                    ))
                except (ValueError, TypeError) as e:
                    continue

        if not fills:
            continue

        # Convert to sequence
        seq = fills_to_sequences(
            fills=fills,
            address=fpath.stem,
            min_trades=10,
        )
        if seq is not None:
            sequences.append(seq)

    print(f"    Loaded {len(sequences)} valid sequences")
    total_trades = sum(s.length for s in sequences)
    print(f"    Total trades: {total_trades}")
    return sequences


def train_lstm(
    sequences: list[TradeSequence],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    seq_length: int = 32,
    hidden_dim: int = 128,
    num_layers: int = 2,
    device: str = "auto",
    output_dir: str = "checkpoints",
    save: bool = True,
) -> dict:
    """Train an LSTM model."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[*] Using device: {device}")

    # Create sequences
    print("[*] Creating training sequences...")
    X, y_action, y_price, y_size = create_sequences_for_training(
        sequences, seq_length=seq_length
    )
    print(f"    {len(X)} samples, {X.shape[2]} features, seq_len={seq_length}")

    if len(X) == 0:
        print("[!] Not enough data to create sequences!")
        return {}

    # Train/val split
    split = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = TensorDataset(
        torch.FloatTensor(X[train_idx]),
        torch.LongTensor(y_action[train_idx]),
        torch.FloatTensor(y_price[train_idx]),
        torch.FloatTensor(y_size[train_idx]),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X[val_idx]),
        torch.LongTensor(y_action[val_idx]),
        torch.FloatTensor(y_price[val_idx]),
        torch.FloatTensor(y_size[val_idx]),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    model = TradeLSTM(
        input_dim=X.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print(f"[*] Starting training ({epochs} epochs)...")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        scheduler.step(val_metrics["loss"])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:3d}/{epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"train_acc={train_metrics['accuracy']:.3f}  "
                f"val_acc={val_metrics['accuracy']:.3f}  "
                f"lr={lr_now:.2e}"
            )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            if save:
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                torch.save(model.state_dict(), output_path / "lstm_best.pt")

    # Final metrics
    print("\n[*] Final evaluation on validation set:")
    final_metrics = evaluate(model, val_loader, device)
    print(f"    Loss: {final_metrics['loss']:.4f}")
    print(f"    Accuracy: {final_metrics['accuracy']:.3f}")

    history["best_val_loss"] = best_val_loss
    history["final_val_accuracy"] = final_metrics["accuracy"]
    history["device"] = str(device)

    if save:
        with open(Path(output_dir) / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    return history


def train_markov_baseline(sequences: list[TradeSequence], order: int = 2) -> dict:
    """Train and evaluate Markov chain baseline."""
    print(f"\n[*] Training Markov-{order} baseline...")

    # Use 80/20 split
    np.random.shuffle(sequences)
    split = int(0.8 * len(sequences))
    train_seqs, val_seqs = sequences[:split], sequences[split:]

    model = MarkovModel(order=order)
    model.fit(train_seqs)

    results = model.evaluate(val_seqs)
    print(f"    Accuracy: {results['accuracy']:.4f}  ({results['correct']}/{results['total']})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train next-trade prediction model")
    parser.add_argument("--model", choices=["lstm", "markov", "all"], default="all")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-length", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--markov-order", type=int, default=2)
    args = parser.parse_args()

    # Load data
    sequences = load_data(args.data_dir)
    if not sequences:
        return

    if args.model in ("markov", "all"):
        train_markov_baseline(sequences, order=args.markov_order)

    if args.model in ("lstm", "all"):
        train_lstm(
            sequences,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seq_length=args.seq_length,
            hidden_dim=args.hidden_dim,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
