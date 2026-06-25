"""
Markov Chain baseline model for next-trade prediction.

A simple n-th order Markov chain that learns transition probabilities
between action states. This serves as a baseline to beat with the LSTM model.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

from ..data.preprocess import TradeSequence, NUM_ACTIONS


@dataclass
class MarkovModel:
    """N-th order Markov chain for action prediction."""
    order: int = 1
    _transitions: dict = field(default_factory=lambda: defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)))
    _prior: np.ndarray = field(default_factory=lambda: np.ones(NUM_ACTIONS, dtype=np.float64))

    def fit(self, sequences: list[TradeSequence]):
        """Fit Markov chain on action sequences."""
        self._transitions.clear()

        for seq in sequences:
            actions = seq.actions
            for i in range(self.order, len(actions)):
                state = tuple(actions[i - self.order:i].tolist())
                next_action = actions[i]
                self._transitions[state][next_action] += 1.0

        # Add-1 smoothing
        for state in self._transitions:
            self._transitions[state] += 1.0

    def predict_next(self, history: np.ndarray) -> int:
        """Predict next action given history.

        Args:
            history: (order,) array of recent actions.

        Returns:
            Predicted action label.
        """
        state = tuple(history[-self.order:].tolist())
        if state in self._transitions:
            probs = self._transitions[state]
        else:
            probs = self._prior
        return int(np.argmax(probs))

    def predict_proba(self, history: np.ndarray) -> np.ndarray:
        """Get probability distribution over next actions."""
        state = tuple(history[-self.order:].tolist())
        if state in self._transitions:
            probs = self._transitions[state]
        else:
            probs = self._prior
        return probs / probs.sum()

    def evaluate(self, sequences: list[TradeSequence]) -> dict:
        """Evaluate Markov model on held-out sequences."""
        correct = 0
        total = 0
        for seq in sequences:
            actions = seq.actions
            for i in range(self.order, len(actions)):
                pred = self.predict_next(actions[:i])
                if pred == actions[i]:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "order": self.order,
        }
