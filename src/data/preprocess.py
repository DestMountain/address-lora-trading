"""
T×P×I sequence preprocessing for address-level behavior modeling.

Each address's trade history is converted into a sequence of tokens,
where each token represents (time_delta, action, price_change).
"""

import numpy as np
import pandas as pd
from typing import Literal
from dataclasses import dataclass

from .hyperliquid import Fill


ACTION_LONG = 0   # opening a long position
ACTION_SHORT = 1  # opening a short position
ACTION_CLOSE_LONG = 2
ACTION_CLOSE_SHORT = 3
ACTION_HOLD = 4   # position maintained (for padding/virtual steps)
NUM_ACTIONS = 5


@dataclass
class TradeSequence:
    """A single address's T×P×I sequence."""
    address: str
    timestamps: np.ndarray      # (seq_len,) int64 ms timestamps
    actions: np.ndarray         # (seq_len,) int32 action labels
    price_deltas: np.ndarray    # (seq_len,) float32 price change ratios
    sizes: np.ndarray           # (seq_len,) float32 trade sizes
    pnls: np.ndarray            # (seq_len,) float32 realized PnL

    @property
    def length(self) -> int:
        return len(self.timestamps)

    def __repr__(self):
        return f"TradeSequence(address={self.address[:8]}..., len={self.length})"


def classify_action(fill: Fill, prev_position: float) -> int:
    """Classify the action type based on fill side and position change.

    Args:
        fill: Current fill.
        prev_position: Position size before this fill.

    Returns:
        Action label (0-4).
    """
    side = fill.side  # 'A' = ask/sell, 'B' = bid/buy
    pos_before = abs(prev_position)
    sz = fill.sz

    if prev_position >= 0 and side == 'B':
        return ACTION_LONG      # opening/increasing long
    elif prev_position <= 0 and side == 'A':
        return ACTION_SHORT     # opening/increasing short
    elif prev_position > 0 and side == 'A':
        return ACTION_CLOSE_LONG
    elif prev_position < 0 and side == 'B':
        return ACTION_CLOSE_SHORT
    else:
        # Fallback based on dir field
        if fill.dir in ("Open Long", "Increase Long"):
            return ACTION_LONG
        elif fill.dir in ("Open Short", "Increase Short"):
            return ACTION_SHORT
        elif fill.dir in ("Close Long", "Reduce Long"):
            return ACTION_CLOSE_LONG
        elif fill.dir in ("Close Short", "Reduce Short"):
            return ACTION_CLOSE_SHORT
        elif fill.dir == "Settlement":
            return ACTION_CLOSE_LONG if prev_position > 0 else ACTION_CLOSE_SHORT
        return ACTION_HOLD


def fills_to_sequences(
    fills: list[Fill],
    address: str,
    min_trades: int = 10,
) -> TradeSequence | None:
    """Convert a list of fills to a T×P×I sequence.

    Args:
        fills: List of fills sorted by time (ascending).
        address: Wallet address.
        min_trades: Minimum number of trades to include.

    Returns:
        TradeSequence or None if too few trades.
    """
    if len(fills) < min_trades:
        return None

    # Sort by time
    fills_sorted = sorted(fills, key=lambda f: f.time)

    n = len(fills_sorted)
    timestamps = np.zeros(n, dtype=np.int64)
    actions = np.zeros(n, dtype=np.int32)
    price_deltas = np.zeros(n, dtype=np.float32)
    sizes = np.zeros(n, dtype=np.float32)
    pnls = np.zeros(n, dtype=np.float32)

    prev_price = None
    prev_position = 0.0

    for i, f in enumerate(fills_sorted):
        timestamps[i] = f.time
        sizes[i] = f.sz * (1 if f.side == 'B' else -1)  # positive for buy, negative for sell

        # Action classification
        actions[i] = classify_action(f, prev_position)

        # Track position
        if f.side == 'B':
            prev_position += f.sz
        elif f.side == 'A':
            prev_position -= f.sz

        # Price change ratio (relative to last trade)
        px = f.px
        if prev_price is not None and prev_price > 0:
            price_deltas[i] = (px - prev_price) / prev_price
        else:
            price_deltas[i] = 0.0
        prev_price = px

        # PnL
        pnls[i] = float(f.closed_pnl)

    return TradeSequence(
        address=address,
        timestamps=timestamps,
        actions=actions,
        price_deltas=price_deltas,
        sizes=sizes,
        pnls=pnls,
    )


def make_time_features(timestamps: np.ndarray) -> np.ndarray:
    """Create time-based features from timestamps.

    Returns:
        (seq_len, 3) array: [time_delta_seconds, hour_of_day, day_of_week]
    """
    deltas = np.zeros_like(timestamps, dtype=np.float32)
    deltas[1:] = (timestamps[1:] - timestamps[:-1]).astype(np.float32) / 1000.0  # ms → s

    # Time-of-day and day-of-week (normalized)
    seconds = timestamps.astype(np.float32) / 1000.0
    hours = (seconds % 86400) / 86400.0       # 0-1
    days = (seconds // 86400) % 7 / 7.0       # 0-1

    return np.column_stack([deltas, hours, days])


def create_sequences_for_training(
    sequences: list[TradeSequence],
    seq_length: int = 32,
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding-window training samples from trade sequences.

    For each address, creates overlapping windows of `seq_length` trades,
    predicting the next action and price delta.

    Returns:
        X: (n_samples, seq_length, feature_dim) input features
        y_action: (n_samples,) next action labels
        y_price: (n_samples,) next price delta
    """
    all_X = []
    all_y_action = []
    all_y_price = []

    for seq in sequences:
        n = seq.length
        if n < seq_length + 1:
            continue

        # Build features: [time_features, action_onehot, price_delta, size]
        time_feat = make_time_features(seq.timestamps)  # (n, 3)
        action_onehot = np.eye(NUM_ACTIONS)[seq.actions]  # (n, 5)
        price_delta = seq.price_deltas.reshape(-1, 1)     # (n, 1)
        size_feat = seq.sizes.reshape(-1, 1)              # (n, 1)

        features = np.concatenate([time_feat, action_onehot, price_delta, size_feat], axis=1)
        # feature_dim = 3 + 5 + 1 + 1 = 10

        for i in range(0, n - seq_length, stride):
            X = features[i:i + seq_length]               # (seq_length, feature_dim)
            y_action = seq.actions[i + seq_length]        # next action
            y_price = seq.price_deltas[i + seq_length]    # next price delta

            all_X.append(X)
            all_y_action.append(y_action)
            all_y_price.append(y_price)

    if not all_X:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(all_X, dtype=np.float32),
        np.array(all_y_action, dtype=np.int64),
        np.array(all_y_price, dtype=np.float32),
    )
