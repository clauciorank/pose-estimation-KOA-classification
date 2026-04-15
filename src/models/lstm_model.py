"""
LSTM and Bi-LSTM classifiers for gait cycle sequences.

Input:  (batch, 101, 4)  — normalised cycle [R_knee, L_knee, R_hip, L_hip]
Output: (batch, 3)       — logits for [NM, KOA, PD]

Both architectures share the same hyper-parameters and can be selected
via the `bidirectional` flag.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaitLSTM(nn.Module):
    """Stacked (Bi-)LSTM classifier for normalised gait cycles.

    Architecture:
        Input → LSTM layer 1 → Dropout → LSTM layer 2 → Dropout
              → last-step hidden → Linear → 3-class logits

    Args:
        input_size:    number of input features per timestep (default 4)
        hidden_size:   hidden units per LSTM layer (default 64)
        num_layers:    number of stacked LSTM layers (default 2)
        num_classes:   output classes (default 3)
        dropout:       dropout between layers (default 0.3)
        bidirectional: if True builds a Bi-LSTM (default False)
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        directions         = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        out, _ = self.lstm(x)          # (batch, seq_len, hidden * directions)
        last   = out[:, -1, :]         # last timestep
        last   = self.dropout(last)
        return self.fc(last)


def build_lstm(
    input_size: int = 4,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_classes: int = 3,
    dropout: float = 0.3,
) -> GaitLSTM:
    return GaitLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=False,
    )


def build_bilstm(
    input_size: int = 4,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_classes: int = 3,
    dropout: float = 0.3,
) -> GaitLSTM:
    return GaitLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=True,
    )