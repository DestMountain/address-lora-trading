"""
LSTM-based sequence model for next-trade prediction.

Architecture:
- Input: (batch, seq_len, feature_dim) T×P×I features
- LSTM layers
- Two heads: action classifier + price delta regressor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..data.preprocess import NUM_ACTIONS


class TradeLSTM(nn.Module):
    """LSTM model for next-trade prediction.

    Predicts both the next action (classification) and
    the next price delta (regression).
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS),
        )

        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) input features.
            return_hidden: If True, also return hidden state.

        Returns:
            action_logits: (batch, num_actions)
            price_pred: (batch, 1)
            (optional) hidden: (batch, hidden_dim)
        """
        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_dim)

        # Two heads
        action_logits = self.action_head(last_hidden)
        price_pred = self.price_head(last_hidden)
        size_pred = self.size_head(last_hidden)

        if return_hidden:
            return action_logits, price_pred, size_pred, last_hidden
        return action_logits, price_pred, size_pred

    def predict_action(self, x: torch.Tensor) -> torch.Tensor:
        """Get action predictions (argmax)."""
        logits, _, _ = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        logits, _, _ = self.forward(x)
        return F.softmax(logits, dim=-1)


class AddressLoRA(nn.Module):
    """Per-address LoRA adapter wrapping a base TradeLSTM.

    Implements LoRA-style low-rank adaptation for each address.
    Only the LoRA matrices are trained per address; the base model is frozen.
    """

    def __init__(self, base_model: TradeLSTM, rank: int = 8):
        super().__init__()
        self.base_model = base_model
        self.rank = rank

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        # LoRA matrices for the LSTM hidden projection
        hidden_dim = base_model.hidden_dim
        self.lora_A = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, hidden_dim) * 0.01)

        # LoRA for the action head
        self.lora_action_A = nn.Parameter(torch.randn(NUM_ACTIONS, rank) * 0.01)
        self.lora_action_B = nn.Parameter(torch.randn(rank, NUM_ACTIONS) * 0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with LoRA adaptation."""
        # Base model forward
        _, _, _, hidden = self.base_model.forward(x, return_hidden=True)

        # LoRA adaptation on hidden state
        lora_delta = hidden @ self.lora_A @ self.lora_B
        adapted_hidden = hidden + lora_delta

        # Action head with LoRA
        action_logits = (
            self.base_model.action_head(adapted_hidden)
            + adapted_hidden @ self.lora_action_A @ self.lora_action_B
        )

        # Price head (no LoRA - just use adapted hidden)
        price_pred = self.base_model.price_head(adapted_hidden)
        size_pred = self.base_model.size_head(adapted_hidden)

        return action_logits, price_pred, size_pred


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 0.5,  # weight for action loss vs price loss
) -> dict[str, float]:
    """Train for one epoch.

    Loss = alpha * CrossEntropy(action) + (1-alpha) * MSE(price + size)
    """
    model.train()
    total_loss = 0.0
    total_action_loss = 0.0
    total_price_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y_action, batch_y_price, batch_y_size in dataloader:
        batch_x = batch_x.to(device)
        batch_y_action = batch_y_action.to(device)
        batch_y_price = batch_y_price.to(device).float()
        batch_y_size = batch_y_size.to(device).float()

        optimizer.zero_grad()

        logits, price_pred, size_pred = model(batch_x)

        action_loss = F.cross_entropy(logits, batch_y_action)
        price_loss = F.mse_loss(price_pred.squeeze(), batch_y_price)
        size_loss = F.mse_loss(size_pred.squeeze(), batch_y_size)
        loss = alpha * action_loss + (1 - alpha) / 2 * price_loss + (1 - alpha) / 2 * size_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_action_loss += action_loss.item()
        total_price_loss += price_loss.item()

        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y_action).sum().item()
        total += batch_y_action.size(0)

    return {
        "loss": total_loss / len(dataloader),
        "action_loss": total_action_loss / len(dataloader),
        "price_loss": total_price_loss / len(dataloader),
        "accuracy": correct / total,
    }


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float = 0.5,
) -> dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    price_errors = []

    with torch.no_grad():
        for batch_x, batch_y_action, batch_y_price, batch_y_size in dataloader:
            batch_x = batch_x.to(device)
            batch_y_action = batch_y_action.to(device)
            batch_y_price = batch_y_price.to(device).float()
            batch_y_size = batch_y_size.to(device).float()

            logits, price_pred, size_pred = model(batch_x)

            action_loss = F.cross_entropy(logits, batch_y_action)
            price_loss = F.mse_loss(price_pred.squeeze(), batch_y_price)
            size_loss = F.mse_loss(size_pred.squeeze(), batch_y_size)
            loss = alpha * action_loss + (1 - alpha) / 2 * price_loss + (1 - alpha) / 2 * size_loss

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y_action).sum().item()
            total += batch_y_action.size(0)

            price_errors.extend(
                (price_pred.squeeze() - batch_y_price).abs().cpu().numpy().tolist()
            )

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "mae": np.mean(price_errors) if price_errors else 0.0,
    }
